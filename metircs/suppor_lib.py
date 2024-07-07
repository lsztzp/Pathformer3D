from pathlib import Path

import matplotlib.cm as cm
import matplotlib
import numpy as np
import torch
import math
import os
import matplotlib.pyplot as plt
import cv2
import matplotlib.image as mpimg

pi = math.pi


def rotate_images(input_path, output_path):
    """Rotate 360-degree images"""
    for _, _, files in os.walk(input_path):
        for name in files:
            for i in range(6):
                angle = str(-180 + i * 60)
                # execute rotation cmd: ffmpeg -i input.png  -vf v360=e:e:yaw=angle output.png
                cmd = 'ffmpeg -i ' + input_path + name + ' -vf v360=e:e:yaw=' + angle + ' ' + \
                      output_path + name.split('.')[0] + '_' + str(i) + '.png'
                os.system(cmd)


def sphere2plane(sphere_cord, height_width=None):
    """ input:  (lat, lon) shape = (n, 2)
        output: (x, y) shape = (n, 2) """
    lat, lon = sphere_cord[:, 0], sphere_cord[:, 1]
    if height_width is None:
        y = (lat + 90) / 180
        x = (lon + 180) / 360
    else:
        y = (lat + 90) / 180 * height_width[0]
        x = (lon + 180) / 360 * height_width[1]
    return torch.cat((y.view(-1, 1), x.view(-1, 1)), 1)


def plane2sphere(plane_cord, height_width=None):
    """ input:  (x, y) shape = (n, 2)
        output: (lat, lon) shape = (n, 2) """
    y, x = plane_cord[:, 0], plane_cord[:, 1]
    if (height_width is None) & (torch.any(plane_cord <= 1).item()):
        lat = (y - 0.5) * 180
        lon = (x - 0.5) * 360
    else:
        lat = (y / height_width[0] - 0.5) * 180
        lon = (x / height_width[1] - 0.5) * 360
    return torch.cat((lat.view(-1, 1), lon.view(-1, 1)), 1)


def sphere2xyz(shpere_cord):
    """ input:  (lat, lon) shape = (n, 2)
        output: (x, y, z) shape = (n, 3) """
    lat, lon = shpere_cord[:, 0], shpere_cord[:, 1]
    lat = lat / 180 * pi
    lon = lon / 180 * pi
    x = torch.cos(lat) * torch.cos(lon)
    y = torch.cos(lat) * torch.sin(lon)
    z = torch.sin(lat)
    return torch.cat((x.view(-1, 1), y.view(-1, 1), z.view(-1, 1)), 1)


def xyz2sphere(threeD_cord):
    """ input: (x, y, z) shape = (n, 3)
        output: (lat, lon) shape = (n, 2) """
    x, y, z = threeD_cord[:, 0], threeD_cord[:, 1], threeD_cord[:, 2]
    lon = torch.atan2(y, x)
    lat = torch.atan2(z, torch.sqrt(x ** 2 + y ** 2))
    lat = lat / pi * 180
    lon = lon / pi * 180
    return torch.cat((lat.view(-1, 1), lon.view(-1, 1)), 1)


def xyz2plane(threeD_cord, height_width=None):
    """ input: (x, y, z) shape = (n, 3)
        output: (x, y) shape = (n, 2) """
    sphere_cords = xyz2sphere(threeD_cord)
    plane_cors = sphere2plane(sphere_cords, height_width)
    return plane_cors


def plot_scanpaths(scanpaths, img_path, save_path="", img_height=512, img_witdth=1024):
    # Plot predicted scanpaths
    # this code is on the basis of ScanGAN https://github.com/DaniMS-ZGZ/ScanGAN360/

    image = cv2.resize(matplotlib.image.imread(img_path), (img_witdth, img_height))
    image_name = img_path.split('/')[-1].split('.')[0]


    plt.imshow(image)

    points_x = scanpaths[:, 1]
    points_y = scanpaths[:, 0]

    colors = cm.rainbow(np.linspace(0, 1, len(points_x)))

    previous_point = None
    for num, x, y, c in zip(range(0, len(points_x)), points_x, points_y, colors):
        x *= img_witdth
        y *= img_height
        markersize = 14.
        linewidth = 6.
        if previous_point is not None:
            if abs(previous_point[0] - x) < (img_witdth / 2):
                plt.plot([x, previous_point[0]], [y, previous_point[1]], color='blue', linewidth=linewidth,
                         alpha=0.35)
            else:
                h_diff = (y - previous_point[1]) / 2
                if x > previous_point[0]:  # X is on the right, Previous is on the Left
                    plt.plot([previous_point[0], 0],
                             [previous_point[1], previous_point[1] + h_diff], color='blue',
                             linewidth=linewidth, alpha=0.35)
                    plt.plot([img_witdth, x], [previous_point[1] + h_diff, y],
                             color='blue', linewidth=linewidth, alpha=0.35)
                else:
                    plt.plot([previous_point[0], img_witdth],
                             [previous_point[1], previous_point[1] + h_diff], color='blue',
                             linewidth=linewidth, alpha=0.35)
                    plt.plot([0, x], [previous_point[1] + h_diff, y], color='blue', linewidth=linewidth,
                             alpha=0.35)
        previous_point = [x, y]
        plt.plot(x, y, marker='o', markersize=markersize, color=c, alpha=.8)
    plt.axis('off')
    # plt.show(bbox_inches='tight', pad_inches=-0.1)
    # plt.margins(-0.1, -0.1)
    plt.margins(0, 0)
    if not save_path:
        plt.show(bbox_inches='tight', pad_inches=-0.1)
    else:
        plt.savefig(str(save_path), bbox_inches='tight', pad_inches=-0.1, dpi=1000)
    plt.cla()
