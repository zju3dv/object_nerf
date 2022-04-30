import matplotlib.pyplot as plt
import numpy as np
import argparse
import cv2


def colored_data(x, cmap="jet", d_min=None, d_max=None):
    if d_min is None:
        d_min = np.min(x)
    if d_max is None:
        d_max = np.max(x)
    print(np.min(x), np.max(x))
    x_relative = (x - d_min) / (d_max - d_min)
    cmap_ = plt.cm.get_cmap(cmap)
    return (255 * cmap_(x_relative)[:, :, :3]).astype(np.uint8)  # H, W, C


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="path to image")
    parser.add_argument("image_file", type=str)
    args = parser.parse_args()
    img = cv2.imread(args.image_file, cv2.IMREAD_ANYDEPTH)

    # print(np.unique(img))
    # img[img!=48] = 0
    # plt.imshow(img, cmap='jet')
    # plt.colorbar()
    # plt.imsave('depth_vis.png', img, cmap='viridis')
    # plt.show()

    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDBLCLK:
            # print(x, y)
            global img
            # print(img.shape)
            print("instance id", img[y, x])

    cv2.namedWindow("image")
    cv2.setMouseCallback("image", mouse_callback)

    color_img = colored_data(img)
    while 1:
        cv2.imshow("image", color_img)
        k = cv2.waitKey(20) & 0xFF
        if k == 27:
            break
