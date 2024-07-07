import copy
import json
import os
import random
from pathlib import Path
import numpy as np
import torch
from datasets.utils import rotate_scanpath
from metircs import suppor_lib

rotate = False  # 是否执行原图旋转到指定目录
IMAGE_PATH = Path("/data/qmengyu/01-Datasets/01-ScanPath-Dataset/salient360/images/")

GAZE_ORIGIN_PATH = Path("/data/qmengyu/01-Datasets/01-ScanPath-Dataset/salient360/Raw/Images/H/Scanpaths/")
GAZE_PATH = Path("/data/qmengyu/01-Datasets/01-ScanPath-Dataset/salient360/fixations/")

if not GAZE_PATH.exists():
    GAZE_PATH.mkdir(parents=True)


def rotate_images(input_path, output_path):
    """Rotate 360-degree images"""
    "水平旋转 -180° ~ 180° "
    for _, _, files in os.walk(input_path):
        for name in files:
            for i in range(6):
                angle = str(-180 + i * 60)
                # execute rotation cmd: ffmpeg -i input.png  -vf v360=e:e:yaw=angle output.png
                cmd = 'ffmpeg -i ' + input_path + name + ' -vf v360=e:e:yaw=' + angle + ' ' + \
                      output_path + name.split('.')[0] + '_' + str(i) + '.jpg'
                os.system(cmd)


if __name__ == '__main__':

    if rotate:  # 图像旋转
        rotate_images(str(IMAGE_PATH) + '/', str(IMAGE_PATH) + '/')

    images_paths = [file_path for file_path in IMAGE_PATH.glob("*.jpg") if not '_' in file_path.stem[4:]]

    # random.shuffle(images_paths)  #全部验证
    # # cnt = int(len(images_paths) * 0.8)   # 对于Salient360, 全部用于验证，这里暂时仿照AOI 80%用于训练，20%用于验证
    # cnt = int(len(images_paths) * 0)

    images_paths.sort()
    cnt=60

    train_images_paths = images_paths[:cnt]
    val_images_paths = images_paths[cnt:]
    train_images_paths.sort()
    val_images_paths.sort()

    data_dict = {
        "all": [],  # 当数据集整体用于测试时使用
        "train": [],  # 使用数据增强
        "test": []
    }

    for phase, phase_images_paths in zip(["train", "test"], [train_images_paths, val_images_paths]):
        print(phase)

        for image_path in phase_images_paths:
            print(image_path)
            image_name = image_path.stem
            its_index = image_name.split('_')[0].split('P')[1]
            gaze_path = GAZE_ORIGIN_PATH / ('Hscanpath_' + its_index + '.txt')

            f = open(gaze_path, "r")
            index = 0
            scanpaths, x, y = [], [], []
            for row in f:
                if index == 0:  # pass the 1s line
                    index += 1
                    continue
                # print(row_id, lon, lat, file_name)
                row_id = int(row.split(',')[0])
                lon = float(row.split(',')[1])
                lat = float(row.split(',')[2])

                if (row_id + 1) // 100 == 1:  # next user
                    _gaze = np.concatenate(
                        (np.array(y).reshape(-1, 1), np.array(x).reshape(-1, 1)), axis=1)
                    scanpaths.append(
                        suppor_lib.plane2sphere(torch.from_numpy(_gaze)).numpy())  # 这里不需要传递高宽参数吗
                    x, y = [], []
                elif (row_id + 4) % 4 == 0:  # sampling 25 points from 100 points
                    x.append(lon)
                    y.append(lat)
                else:
                    continue

            scanpaths = np.array(scanpaths)
            torch.save(torch.from_numpy(scanpaths), GAZE_PATH / (image_path.stem + '.pck'))

            origin_data = {"image_path": str(IMAGE_PATH / (str(image_path.stem) + '.jpg')),
                           "scanpaths": str(GAZE_PATH / (image_path.stem + '.pck'))}

            if phase != "train":
                data_dict[phase].append(origin_data)
            data_dict["all"].append(origin_data)
            for rotation_id in range(6):
                scanpaths_copy = copy.deepcopy(scanpaths)
                # f.write(str(image_path.stem) + '_' + str(rotation_id) + '.png' + "\n")
                rotation_angle = rotation_id * 60 - 180  # 【-180， 120】
                gaze_sphere = torch.zeros(scanpaths.shape[0], scanpaths.shape[1], 2)

                for scanpath_id in range(0, scanpaths.shape[0]):
                    gaze_sphere[scanpath_id] = torch.from_numpy(
                        rotate_scanpath(scanpaths_copy[scanpath_id], rotation_angle))

                torch.save(gaze_sphere, GAZE_PATH / (image_path.stem + '_' + str(rotation_id) + '.pck'))
                if phase == "train":

                    for scanpath_id in range(0, scanpaths.shape[0]):
                        rotated_data_index = {
                            "image_path": str(IMAGE_PATH / (str(image_path.stem) + '_' + str(rotation_id) + '.jpg')),
                            "scanpaths": str(GAZE_PATH / (image_path.stem + '_' + str(rotation_id) + '.pck')),
                            "index": scanpath_id
                        }
                        data_dict[phase].append(rotated_data_index)

    tf = open(IMAGE_PATH.parent / "dataset.json", "w")
    json.dump(data_dict, tf)
    tf.close()
