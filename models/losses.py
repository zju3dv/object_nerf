import torch
from torch import nn


class OpacityLoss(nn.Module):
    def __init__(self, coef=1):
        super().__init__()
        self.coef = coef
        self.loss = nn.MSELoss(reduction="none")

    def forward(self, inputs, batch):
        valid_mask = batch["valid_mask"].view(-1)
        if valid_mask.sum() == 0:  # skip when mask is empty
            return None
        instance_mask = batch["instance_mask"].view(-1)[valid_mask]
        instance_mask_weight = batch["instance_mask_weight"].view(-1)[valid_mask]
        loss = (
            self.loss(
                torch.clamp(inputs["opacity_instance_coarse"][valid_mask], 0, 1),
                instance_mask.float(),
            )
            * instance_mask_weight
        ).mean()
        if "opacity_instance_fine" in inputs:
            loss += (
                self.loss(
                    torch.clamp(inputs["opacity_instance_fine"][valid_mask], 0, 1),
                    instance_mask.float(),
                )
                * instance_mask_weight
            ).mean()
        return self.coef * loss


class DepthLoss(nn.Module):
    def __init__(self, coef=1, instance_only=False):
        super().__init__()
        self.coef = coef
        self.loss = nn.MSELoss(reduction="none")
        self.instance_only = instance_only

    def forward(self, inputs, batch):
        targets = batch["depths"].view(-1)
        if (targets > 0).sum() == 0:
            return None
        mask = (batch["valid_mask"] * (targets > 0)).view(-1)  # (H*W)
        if self.instance_only:
            mask = mask * batch["instance_mask"].view(-1)
            if mask.sum() == 0:  # skip when instance mask is empty
                return None
            instance_mask_weight = batch["instance_mask_weight"].view(-1)[mask]
            loss = (
                self.loss(inputs["depth_instance_coarse"][mask], targets[mask])
                * instance_mask_weight
            ).mean()
            if "depth_instance_fine" in inputs:
                loss += (
                    self.loss(inputs["depth_instance_fine"][mask], targets[mask])
                    * instance_mask_weight
                ).mean()
        else:
            loss = self.loss(inputs["depth_coarse"][mask], targets[mask]).mean()
            if "rgb_fine" in inputs:
                loss += self.loss(inputs["depth_fine"][mask], targets[mask]).mean()
        return self.coef * loss


class ColorLoss(nn.Module):
    def __init__(self, coef=1, instance_only=False):
        super().__init__()
        self.coef = coef
        self.loss = nn.MSELoss(reduction="none")
        self.instance_only = instance_only

    def forward(self, inputs, batch):
        targets = batch["rgbs"].view(-1, 3)
        mask = batch["valid_mask"].view(-1, 1).repeat(1, 3)  # (H*W, 3)
        if self.instance_only:
            mask = mask * batch["instance_mask"].view(-1, 1).repeat(1, 3)
            if mask.sum() == 0:  # skip when instance mask is empty
                return None
            instance_mask_weight = (
                batch["instance_mask_weight"].view(-1, 1).repeat(1, 3)[mask]
            )
            loss = (
                self.loss(inputs["rgb_instance_coarse"][mask], targets[mask])
                * instance_mask_weight
            ).mean()
            if "rgb_instance_fine" in inputs:
                loss += (
                    self.loss(inputs["rgb_instance_fine"][mask], targets[mask])
                    * instance_mask_weight
                ).mean()
        else:
            loss = self.loss(inputs["rgb_coarse"][mask], targets[mask]).mean()
            if "rgb_fine" in inputs:
                loss += self.loss(inputs["rgb_fine"][mask], targets[mask]).mean()

        return self.coef * loss


class TotalLoss(nn.Module):
    def __init__(self, conf):
        super().__init__()
        self.conf = conf
        self.color_loss = ColorLoss(self.conf["color_loss_weight"])
        self.depth_loss = DepthLoss(self.conf["depth_loss_weight"])
        self.opacity_loss = OpacityLoss(self.conf["opacity_loss_weight"])
        self.instance_color_loss = ColorLoss(
            self.conf["instance_color_loss_weight"], True
        )
        self.instance_depth_loss = DepthLoss(
            self.conf["instance_depth_loss_weight"], True
        )

    def forward(self, inputs, batch, epoch=-1):
        loss_dict = dict()

        loss_dict["color_loss"] = self.color_loss(inputs, batch)
        loss_dict["depth_loss"] = self.depth_loss(inputs, batch)
        loss_dict["opacity_loss"] = self.opacity_loss(inputs, batch)
        loss_dict["instance_color_loss"] = self.instance_color_loss(inputs, batch)
        loss_dict["instance_depth_loss"] = self.instance_depth_loss(inputs, batch)

        # remove unused loss
        loss_dict = {k: v for k, v in loss_dict.items() if v != None}

        loss_sum = sum(list(loss_dict.values()))

        # recover loss to orig scale for comparison
        for k, v in loss_dict.items():
            if f"{k}_weight" in self.conf:
                loss_dict[k] /= self.conf[f"{k}_weight"]

        return loss_sum, loss_dict


def get_loss(config):
    return TotalLoss(config.loss)
