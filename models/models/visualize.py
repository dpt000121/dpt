import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os
import torch

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def vis_feat(feature_tensor, col, raw, save_path, name, colormode=2, margining=1):
    """
    COLORMAP_AUTUMN = 0,
    COLORMAP_BONE = 1,
    COLORMAP_JET = 2,
    COLORMAP_WINTER = 3,
    COLORMAP_RAINBOW = 4,
    COLORMAP_OCEAN = 5,
    COLORMAP_SUMMER = 6,
    COLORMAP_SPRING = 7,
    COLORMAP_COOL = 8,
    COLORMAP_HSV = 9,
    COLORMAP_PINK = 10,
    COLORMAP_HOT = 11
    :param feature_tensor: torch.Tensor [1,c,w,h]
    :param col: col num
    :param raw: raw num
    :param save_path: save path
    :param colormode: cv2.COLORMAP
    :return:None
    """
    import cv2
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    import os
    import time
    time.ctime()
    show_k = col * raw
    f = feature_tensor[0, :show_k, :, :]
    size = f[0, :, :].shape
    f = f.cpu().data.numpy()
    f = (f - np.min(f, axis=(1, 2), keepdims=True)) / (
            np.max(f, axis=(1, 2), keepdims=True) - np.min(f, axis=(1, 2), keepdims=True) + 1e-8) * 255
    for i in range(raw):
        tem = f[i * col, :, :]
        tem = cv2.applyColorMap(np.array(tem, dtype=np.uint8), colormode)
        for j in range(col):
            if not j == 0:
                tem = np.concatenate((tem, np.ones((size[0], margining, 3), dtype=np.uint8) * 255), 1)
                tem2 = cv2.applyColorMap(np.array(f[i * col + j, :, :], dtype=np.uint8), colormode)
                tem = np.concatenate((tem, tem2), 1)
        if i == 0:
            final = tem
        else:
            final = np.concatenate(
                (final, np.ones((margining, size[1] * col + (col - 1) * margining, 3), dtype=np.uint8) * 255), 0)
            final = np.concatenate((final, tem), 0)

    # print(final.shape)
    # if not os.path.exists(save_path):
    #     os.mkdir(save_path)
    # cv2.imwrite(os.path.join(save_path, time.ctime()+'.jpg'), final)
    cv2.imwrite(save_path + '/' + name , final)
    # cv2.imwrite(save_path, final)


def vis_alpha(name, image_path, features, save_path, alpha=0.2, mode=1):
    """
    mode=1  jet
    mode=2  Reds
    mode=3  gist_gray
    """
    image = cv2.imread(image_path)
    mean_feat = torch.mean(features, dim=1, keepdim=True).squeeze(0).permute(1, 2, 0).data.numpy()
    mean_feat = cv2.resize(mean_feat, dsize=image.shape[:-1][::-1], interpolation=cv2.INTER_LANCZOS4)
    plt.imshow(image)
    plt.imshow(mean_feat, alpha=alpha)
    if mode == 1:
        plt.set_cmap(cm.jet)
    elif mode == 2:
        plt.set_cmap(cm.Reds)
    else:
        plt.set_cmap(cm.gist_gray)
    plt.axis('off')

    plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
    plt.margins(0, 0)
    plt.savefig(save_path + name)
    plt.close()


def vis_grid(grid, save_path, grid_size=[50, 50]):
    grid = grid.squeeze().permute(1, 2, 0).detach().numpy()
    assert grid.ndim == 3

    size = grid.shape
    x = np.linspace(-1, 1, size[1])
    y = np.linspace(-1, 1, size[0])
    X, Y = np.meshgrid(x, y)
    Z1 = grid[:, :, 0] + 2  # remove the dashed line
    Z1 = Z1[::-1]  # vertical flip
    Z2 = grid[:, :, 1] + 2
    Z2 = Z2[::-1]  # vertical flip

    plt.figure()
    plt.contour(X, Y, Z1, grid_size[1], colors='k')
    plt.contour(X, Y, Z2, grid_size[0], colors='k')
    plt.xticks(()), plt.yticks(())  # remove x, y ticks
    plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
    plt.margins(0, 0)
    plt.savefig(save_path)