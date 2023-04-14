# Object-NeRF: Data Preparation

## ToyDesk Dataset

You can download the data through this [link](https://www.dropbox.com/s/bdqiv7pc13p6ugp/toydesk_data_full.zip?dl=0).

Please note that the transforms_xxx.json comes with transformation matrix (from camera coordinate to world coordinate) in SLAM / OpenCV format (xyz -> right down forward), thus we need to change to NDC format (xyz -> right up back) when applying to standard NeRF training codes.

## ScanNet Dataset

You can download the pre-processed files through this [link](https://www.dropbox.com/s/k7mxkuone3ucsgd/scannet_object_nerf_data.zip?dl=0).

### Step 1: Generate ScanNet training frames

First, we generate training frames (RGB-D image, instance segmentation) from ScanNet files to NeRF-like format.
For example, to generate `scene00xx_00`, you can run script like that:

```bash
python data_preparation/scannet_sens_reader/convert_to_nerf_style_data.py \
    --input data/scannet/scans/scene00xx_00/ \
    --output data/scannet/processed_scannet_00xx_00 \
    --instance_filt_dir data/scannet/scans/scene00xx_00/instance-filt
```
Please note that the generated transforms_xxx.json comes with transformation matrix (from camera coordinate to world coordinate) in SLAM / OpenCV format (xyz -> right down forward), thus we need to change to NDC format (xyz -> right up back) in our dataloader.

### Step 2: Generate ScanNet scene point clouds

When enabling voxel based representation, we use script from [NPCR](https://github.com/daipengwa/Neural-Point-Cloud-Rendering-via-Multi-Plane-Projection/blob/master/pre_processing/generate_pointclouds_ScanNet.py) to generate point clouds via depth lifting.


### Step 3: Make sure the config file matches data path

We assume the config file as `config/scannet_base_0113.yml`.
    - `root_dir` should be matched to the generated data folder in Step 1.
    - `pcd_path` should be matched to the generated scene point cloud in Step 2.
    - `scans_dir` is pointing to the ScanNet's official unzipped scans.
Note that the bounding box and scene transformation has been stored in `scannet_train_detection_data`, which can also be generated following [VoteNet docs](https://github.com/facebookresearch/votenet/blob/main/scannet/README.md).


For convenience, you can directly download the preprocessed `scannet_train_detection_data` (Step 3) and scene point cloud (Step 2) from this [link](https://zjueducn-my.sharepoint.com/:u:/g/personal/ybbbbt_zju_edu_cn/ETNgkZwpDnxDlXy3ISevnnQBAWENRZ6j0voeqlfvpijr6A?e=nrMFBS).
