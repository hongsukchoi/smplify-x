import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from constant import mano_skeleton

def vis_3d_skeleton(kpt_3d, kpt_3d_vis, filename=None):

    kps_lines = mano_skeleton

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Convert from plt 0-1 RGBA colors to 0-255 BGR colors for opencv.
    cmap = plt.get_cmap('rainbow')
    colors = [cmap(i) for i in np.linspace(0, 1, len(kps_lines) + 2)]
    colors = [np.array((c[2], c[1], c[0])) for c in colors]

    for l in range(len(kps_lines)):
        i1 = kps_lines[l][0]
        i2 = kps_lines[l][1]
        x = np.array([kpt_3d[i1, 0], kpt_3d[i2, 0]])
        y = np.array([kpt_3d[i1, 1], kpt_3d[i2, 1]])
        z = np.array([kpt_3d[i1, 2], kpt_3d[i2, 2]])

        if kpt_3d_vis[i1, 0] > 0 and kpt_3d_vis[i2, 0] > 0:
            ax.plot(x, z, -y, c=colors[l], linewidth=2)
        if kpt_3d_vis[i1, 0] > 0:
            ax.text(kpt_3d[i1, 0], kpt_3d[i1, 2], -
                      kpt_3d[i1, 1], s=str(i1))
            ax.scatter(kpt_3d[i1, 0], kpt_3d[i1, 2], -
                       kpt_3d[i1, 1], c=colors[l], marker='o')
        if kpt_3d_vis[i2, 0] > 0:
            ax.text(kpt_3d[i2, 0], kpt_3d[i2, 2], -
                    kpt_3d[i2, 1], s=str(i2))
            ax.scatter(kpt_3d[i2, 0], kpt_3d[i2, 2], -
                       kpt_3d[i2, 1], c=colors[l], marker='o')

    if filename is None:
        ax.set_title('3D vis')
    else:
        ax.set_title(filename)

    ax.set_xlabel('X Label')
    ax.set_ylabel('Z Label')
    ax.set_zlabel('Y Label')
    ax.legend()

    plt.show()
    cv2.waitKey(0)

    # plt.savefig(filename)
