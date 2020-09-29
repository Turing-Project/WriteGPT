  
import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt
import cv2
import torch
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

pairs = [[i, i + 1] for i in range(16)] + \
                         [[i, i + 1] for i in range(17, 21)] + \
                         [[i, i + 1] for i in range(22, 26)] + \
                         [[i, i + 1] for i in range(36, 41)] + [[41, 36]] + \
                         [[i, i + 1] for i in range(42, 47)] + [[47, 42]] + \
                         [[i, i + 1] for i in range(27, 30)] + \
                         [[i, i + 1] for i in range(31, 35)] + \
                         [[i, i + 1] for i in range(48, 59)] + [[59, 48]] + \
                         [[i, i + 1] for i in range(60, 67)] + [[67, 60]]

def show_joints(img, pts, show_idx=False, pairs=None):

    fig, ax = plt.subplots()
    ax.imshow(img)

    for i in range(pts.shape[0]):
        if pts[i, 2] > 0:
            ax.scatter(pts[i,0], pts[i,1], s=10, c='c', edgecolors='b', linewidth=0.3)
            if show_idx:
                plt.text(pts[i, 0], pts[i, 1], str(i))
            if pairs is not None:
                for p in pairs:
                    ax.plot(pts[p, 0], pts[p, 1], c='b', linewidth=0.3)

    plt.axis('off')
    plt.show()
    plt.close()

def show_joints_heatmap(img, target):

    img = cv2.resize(img, target.shape[1:])
    for i in range(target.shape[0]):
        t = target[i, :, :]
        plt.imshow(img, alpha=0.5)
        plt.imshow(t, alpha=0.5)
        plt.axis('off')
        plt.show()
    plt.close()

def show_joints_boundary(img, target):

    img = cv2.resize(img, target.shape[1:])
    for i in range(target.shape[0]):
        t = target[i, :, :]
        plt.imshow(img, alpha=0.5)
        plt.imshow(t, alpha=0.5)
        plt.axis('off')
        plt.show()
    plt.close()

# def show_joints_3d(img, pts, show_idx=False, pairs=None):
#
#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')
#
#     ax.imshow(img)
#
#     for i in range(pts.size(0)):
#         if pts[i, 2] > 0:
#             ax.scatter(pts[i,0], pts[i,1], pts[i,2], s=5, c='c', edgecolors='b', linewidth=0.3)
#             if show_idx:
#                 plt.text(pts[i, 0], pts[i, 1], str(i))
#
#     plt.axis('off')
#     plt.show()
#     plt.close()
def show_joints_3d(predPts, pairs=None):

    ax = plt.subplot(111, projection='3d')

    view_angle = (-160, 30)
    if predPts.shape[1] > 2:
        ax.scatter(predPts[:, 2], predPts[:, 0], predPts[:, 1], s=5, c='c', marker='o', edgecolors='b', linewidths=0.5)
        # ax_pred.scatter(predPts[0, 2], predPts[0, 0], predPts[0, 1], s=10, c='g', marker='*')
        if pairs is not None:
            for p in pairs:
                ax.plot(predPts[p, 2], predPts[p, 0], predPts[p, 1], c='b',  linewidth=0.5)
    else:
        ax.scatter([0] * predPts.shape[0], predPts[:, 0], predPts[:, 1], s=10, marker='*')
    ax.set_xlabel('z', fontsize=10)
    ax.set_ylabel('x', fontsize=10)
    ax.set_zlabel('y', fontsize=10)
    ax.view_init(*view_angle)
    plt.show()
    plt.close()

def save_plots(config, imgs, ppts_2d, ppts_3d, tpts_2d, tpts_3d, filename, nrows=4, ncols=4):

    # transform images
    mean = np.array(config.DATASET.MEAN, dtype=np.float32)
    std = np.array(config.DATASET.STD, dtype=np.float32)
    imgs = imgs.transpose(0, 2, 3, 1)
    imgs = (imgs * std + mean) * 255.
    imgs = imgs.astype(np.uint8)

    # plot 2d
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15,15))

    cnt = 0
    for i in range(nrows):
        for j in range(ncols):
            # Output a grid of images
            axes[i, j].imshow(imgs[cnt])
            axes[i, j].scatter(ppts_2d[cnt, :, 0]*4, ppts_2d[cnt, :, 1]*4, s=10, c='c', edgecolors='k', linewidth=1)
            axes[i, j].scatter(tpts_2d[cnt, :, 0] * 4, tpts_2d[cnt, :, 1] * 4, s=10, c='r', edgecolors='k', linewidth=1)
            axes[i, j].axis('off')
            if pairs is not None:
                for p in pairs:
                    axes[i, j].plot(ppts_2d[cnt, p, 0] * 4, ppts_2d[cnt, p, 1] * 4, c='b', linewidth=0.5)
                    axes[i, j].plot(tpts_2d[cnt, p, 0] * 4, tpts_2d[cnt, p, 1] * 4, c='r', linewidth=0.5)
            cnt += 1
    plt.savefig(filename + '_2d.png')
    plt.close()

    # plot 3d
    fig = plt.figure(figsize=(15,15))
    for i in range(nrows*ncols):
        ax = fig.add_subplot(nrows, ncols, i+1, projection='3d')
        ax.scatter(ppts_3d[i, :, 2], ppts_3d[i, :, 0], ppts_3d[i, :, 1], s=10, color='b', edgecolor='k', alpha=0.6)
        ax.scatter(tpts_3d[i, :, 2], tpts_3d[i, :, 0], tpts_3d[i, :, 1], s=10, color='r', edgecolor='k', alpha=0.6)
        ax.view_init(elev=205, azim=110)
        # ax.axis('off')
        if pairs is not None:
            for p in pairs:
                ax.plot(ppts_3d[i, p, 2], ppts_3d[i, p, 0], ppts_3d[i, p, 1], c='b', linewidth=1)
                ax.plot(tpts_3d[i, p, 2], tpts_3d[i, p, 0], tpts_3d[i, p, 1], c='r', linewidth=1)
    plt.savefig(filename + '_3d.png')
    plt.close()