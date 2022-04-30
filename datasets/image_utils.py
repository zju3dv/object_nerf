from matplotlib import pyplot as plt
import numpy as np
import cv2
from PIL import Image
import os


def rebalance_mask(mask, fg_weight=None, bg_weight=None):
    if fg_weight is None and bg_weight is None:
        foreground_cnt = max(mask.sum(), 1)
        background_cnt = max((~mask).sum(), 1)
        balanced_weight = np.ones_like(mask).astype(np.float32)
        balanced_weight[mask] = float(background_cnt) / foreground_cnt
        balanced_weight[~mask] = float(foreground_cnt) / background_cnt
    else:
        balanced_weight = np.ones_like(mask).astype(np.float32)
        balanced_weight[mask] = fg_weight
        balanced_weight[~mask] = bg_weight
    # print('fg {} bg {}'.format(foreground_cnt, background_cnt))
    # print(balanced_weight.min(), balanced_weight.max())
    # print(balanced_weight.shape)
    # cv2.normalize(balanced_weight, balanced_weight, 0, 1.0, cv2.NORM_MINMAX)
    # cv2.imshow('img_balanced_weight', balanced_weight)
    # cv2.waitKey(5)
    return balanced_weight


def compute_distance_transfrom_weights(
    mask, uncertain_pixel_distance=15, fg_bg_balance_weight=False
):
    instance_mask = mask
    max_dist = uncertain_pixel_distance
    dt_field = np.zeros_like(instance_mask, dtype=np.uint8)
    dt_field[instance_mask] = 255
    dist1 = cv2.distanceTransform(dt_field, cv2.DIST_L2, 3)

    dt_field_inv = np.zeros_like(instance_mask, dtype=np.uint8)
    dt_field_inv[~instance_mask] = 255
    dist2 = cv2.distanceTransform(dt_field_inv, cv2.DIST_L2, 3)

    dist_combine = np.ones_like(dist1) * max_dist

    dist1[dist1 > max_dist] = max_dist
    dist2[dist2 > max_dist] = max_dist

    dist1_mask = (dist1 < max_dist) * (dist1 > 0)
    dist_combine[dist1_mask] = dist1[dist1_mask]
    disk2_mask = (dist2 < max_dist) * (dist2 > 0)
    dist_combine[disk2_mask] = dist2[disk2_mask]

    cv2.normalize(dist_combine, dist_combine, 0, 1.0, cv2.NORM_MINMAX)

    if fg_bg_balance_weight:
        dist_combine *= rebalance_mask(mask)

    # cv2.normalize(dist_combine, dist_combine, 0, 1, cv2.NORM_MINMAX)
    # cv2.imshow('Distance Transform Image', dist_combine)
    # cv2.waitKey(5)
    return dist_combine
