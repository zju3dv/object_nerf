import os

os.environ["OMP_NUM_THREADS"] = "1"  # noqa
os.environ["MKL_NUM_THREADS"] = "1"  # noqa

from utils.util import get_timestamp, make_source_code_snapshot
import torch
from collections import defaultdict

from torch.utils.data import DataLoader
from datasets import dataset_dict
from omegaconf import OmegaConf

# models
from models.nerf_model import ObjectNeRF
from models.embedding_helper import EmbeddingVoxel, Embedding
from models.rendering import render_rays
from models.code_library import CodeLibrary

# optimizer, scheduler, visualization
from utils import get_optimizer, get_scheduler, get_learning_rate
from utils.train_helper import visualize_val_image

# losses
from models.losses import get_loss

# metrics
from utils.metrics import psnr

# pytorch-lightning
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.loggers import TensorBoardLogger


class ObjectNeRFSystem(LightningModule):
    def __init__(self, config):
        super(ObjectNeRFSystem, self).__init__()
        self.config = config

        self.loss = get_loss(config)

        self.use_voxel_embedding = self.config.model.get("use_voxel_embedding", True)

        if self.use_voxel_embedding:
            self.embedding_xyz = EmbeddingVoxel(
                channels=config.model.N_scn_voxel_size + config.model.N_obj_voxel_size,
                N_freqs=config.model.N_freq_voxel,
                max_voxels=config.model.N_max_voxels,
                dataset_extra_config=config.dataset_extra,
            )
        else:
            self.embedding_xyz = Embedding(3, self.config.model.N_freq_xyz)

        self.embedding_dir = Embedding(3, self.config.model.N_freq_dir)
        self.embeddings = {"xyz": self.embedding_xyz, "dir": self.embedding_dir}

        self.nerf_coarse = ObjectNeRF(self.config.model)
        self.models = {"coarse": self.nerf_coarse}

        if config.model.N_importance > 0:
            self.nerf_fine = ObjectNeRF(self.config.model)
            self.models["fine"] = self.nerf_fine

        self.code_library = CodeLibrary(config.model)

        self.models_to_train = [
            self.models,
            self.code_library,
            self.embedding_xyz,
        ]

    def forward(self, rays, extra=dict()):
        """Do batched inference on rays using chunk."""
        B = rays.shape[0]
        results = defaultdict(list)
        for i in range(0, B, self.config.train.chunk):
            extra_chunk = dict()
            for k, v in extra.items():
                if isinstance(v, torch.Tensor):
                    extra_chunk[k] = v[i : i + self.config.train.chunk]
                else:
                    extra_chunk[k] = v
            rendered_ray_chunks = render_rays(
                models=self.models,
                embeddings=self.embeddings,
                rays=rays[i : i + self.config.train.chunk],
                N_samples=self.config.model.N_samples,
                use_disp=self.config.model.use_disp,
                perturb=self.config.model.perturb,
                noise_std=self.config.model.noise_std,
                N_importance=self.config.model.N_importance,
                chunk=self.config.train.chunk,  # chunk size is effective in val mode
                white_back=self.train_dataset.white_back
                if self.training
                else self.val_dataset.white_back,
                **extra_chunk,
            )

            for k, v in rendered_ray_chunks.items():
                results[k] += [v]

        for k, v in results.items():
            results[k] = torch.cat(v, 0)
        return results

    def setup(self, stage):
        dataset = dataset_dict[self.config.dataset_name]
        kwargs = {
            "img_wh": tuple(self.config.img_wh),
        }
        kwargs["dataset_extra"] = self.config.dataset_extra
        self.train_dataset = dataset(split="train", **kwargs)
        self.val_dataset = dataset(split="val", **kwargs)

    def configure_optimizers(self):
        self.optimizer = get_optimizer(self.config.train, self.models_to_train)
        scheduler = get_scheduler(self.config.train, self.optimizer)
        return [self.optimizer], [scheduler]

    def train_dataloader(self):
        batch_size = self.config.train.batch_size
        return DataLoader(
            self.train_dataset,
            shuffle=True,
            num_workers=6,
            batch_size=batch_size,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            shuffle=False,
            num_workers=4,
            batch_size=1,  # validate one image (H*W rays) at a time
            pin_memory=True,
        )

    def on_epoch_start(self):
        if self.config.train.progressive_train and self.use_voxel_embedding:
            if self.current_epoch > 2:
                self.embedding_xyz.self_pruning_empty_voxels(self.models["fine"])
            if self.current_epoch == 5:
                self.embedding_xyz.voxel_subdivision()

    def training_step(self, batch, batch_nb):
        rays, rgbs = batch["rays"], batch["rgbs"]
        rays = rays.squeeze()  # (H*W, 3)
        rgbs = rgbs.squeeze()  # (H*W, 3)

        # get mask for psnr evaluation
        mask = batch["valid_mask"].view(-1, 1).repeat(1, 3)  # (H*W, 3)
        extra_info = dict()
        extra_info["is_eval"] = False
        # extra_info["instance_mask"] = batch["instance_mask"]
        extra_info["pass_through_mask"] = batch["pass_through_mask"]
        extra_info["rays_in_bbox"] = getattr(
            self.train_dataset, "is_rays_in_bbox", lambda _: False
        )()
        extra_info["frustum_bound_th"] = (
            self.config.model.frustum_bound
            / self.config["dataset_extra"]["scale_factor"]
        )
        extra_info.update(self.code_library(batch))

        results = self(rays, extra_info)
        loss_sum, loss_dict = self.loss(results, batch, self.current_epoch)

        with torch.no_grad():
            typ = "fine" if "rgb_fine" in results else "coarse"
            psnr_ = psnr(results[f"rgb_{typ}"], rgbs, mask)

        self.log("lr", get_learning_rate(self.optimizer))
        self.log("train/loss", loss_sum)
        for k, v in loss_dict.items():
            self.log(f"train/{k}", v)
        self.log("train/psnr", psnr_, prog_bar=True)

        return loss_sum

    def validation_step(self, batch, batch_nb):
        rays, rgbs = batch["rays"], batch["rgbs"]
        # get mask for psnr evaluation
        if "instance_mask" in batch:
            mask = (
                (batch["valid_mask"] * batch["instance_mask"]).view(-1, 1).repeat(1, 3)
            )  # (H*W, 3)
        else:
            mask = None
        rays = rays.squeeze()  # (H*W, 3)
        rgbs = rgbs.squeeze()  # (H*W, 3)
        extra_info = dict()
        extra_info["is_eval"] = True
        extra_info["rays_in_bbox"] = getattr(
            self.val_dataset, "is_rays_in_bbox", lambda _: False
        )()
        extra_info["frustum_bound_th"] = (
            self.config.model.frustum_bound
            / self.config["dataset_extra"]["scale_factor"]
        )
        extra_info.update(self.code_library(batch))
        results = self(rays, extra_info)

        loss_sum, loss_dict = self.loss(results, batch)
        for k, v in loss_dict.items():
            self.log(f"val/{k}", v)
        log = {"val_loss": loss_sum}
        log.update(loss_dict)
        typ = "fine" if "rgb_fine" in results else "coarse"

        if batch_nb == 0:
            stack_image = visualize_val_image(
                self.config.img_wh, batch, results, typ=typ
            )
            self.logger.experiment.add_images(
                "val/GT_pred_depth", stack_image, self.global_step
            )

        psnr_ = psnr(results[f"rgb_{typ}"], rgbs, mask)
        log["val_psnr"] = psnr_

        return log

    def validation_epoch_end(self, outputs):
        mean_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        mean_psnr = torch.stack([x["val_psnr"] for x in outputs]).mean()

        self.log("val/loss", mean_loss)
        self.log("val/psnr", mean_psnr, prog_bar=True)


def main(config):
    exp_name = get_timestamp() + "_" + config.exp_name
    print(f"Start with exp_name: {exp_name}.")
    log_path = f"logs/{exp_name}"
    config["log_path"] = log_path

    system = ObjectNeRFSystem(config)

    checkpoint_callback = ModelCheckpoint(
        dirpath=log_path,
        filename="{epoch:d}",
        monitor="val/psnr",
        mode="max",
        # save_top_k=5,
        save_top_k=-1,
        save_last=True,
        every_n_epochs=1,
        save_on_train_epoch_end=True,
    )

    logger = TensorBoardLogger(save_dir="logs", name=exp_name)

    trainer = Trainer(
        max_epochs=config.train.num_epochs,
        callbacks=[checkpoint_callback],
        resume_from_checkpoint=config.ckpt_path,
        logger=logger,
        enable_model_summary=False,
        gpus=config.train.num_gpus,
        accelerator="ddp" if config.train.num_gpus > 1 else None,
        num_sanity_val_steps=1,
        benchmark=True,
        profiler="simple" if config.train.num_gpus == 1 else None,
        val_check_interval=0.25,
        limit_train_batches=config.train.limit_train_batches,
    )

    make_source_code_snapshot(f"logs/{exp_name}")
    OmegaConf.save(config=config, f=os.path.join(log_path, "run_config_snapshot.yaml"))
    trainer.fit(system)


if __name__ == "__main__":
    conf_cli = OmegaConf.from_cli()
    conf_dataset = OmegaConf.load(conf_cli.dataset_config)
    conf_default = OmegaConf.load("config/default_conf.yml")
    # merge conf with the priority
    conf_merged = OmegaConf.merge(conf_default, conf_dataset, conf_cli)

    print("-" * 40)
    print(OmegaConf.to_yaml(conf_merged))
    print("-" * 40)

    main(config=conf_merged)
