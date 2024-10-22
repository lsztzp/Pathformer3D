import json
import os
import random
from pathlib import Path
from statistics import mean

import numpy as np
import torch
from datasets.utils import rotate_scanpath

rotate = False  # if rotate
IMAGE_PATH = Path("/data/qmengyu/01-Datasets/01-ScanPath-Dataset/AOI/images/")    #saved image dir

GAZE_ORIGIN_PATH = Path(
    "/data/qmengyu/01-Datasets/01-ScanPath-Dataset/AOI/Raw/TIP2021_AOI_Dataset/AOI_salmap&fixation/data_3rdn_1vision/")  #raw scapath path
GAZE_PATH = Path("/data/qmengyu/01-Datasets/01-ScanPath-Dataset/AOI/fixations/")     #saved path


if not GAZE_PATH.exists():
    GAZE_PATH.mkdir(parents=True)

def rotate_images(input_path, output_path):
    """Rotate 360-degree images"""
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

    images_paths = [file_path for file_path in IMAGE_PATH.glob("*.jpg") if not '_' in file_path.stem[-2:]]

    random.shuffle(images_paths)  # 对于AOI数据集，随机选择 20%验证
    cnt = int(len(images_paths) * 0.8)
    train_images_paths = images_paths[:cnt]
    test_images_paths = images_paths[cnt:]

    train_images_paths.sort()
    test_images_paths.sort()

    data_dict = {
        "all": [],
        "train": [],
        "test": []
    }
    length_list = []
    for phase, phase_images_paths in zip(["train", "test"], [train_images_paths, test_images_paths]):
        print(phase)

        for image_path in phase_images_paths:
            print(image_path)

            gaze_path = GAZE_ORIGIN_PATH / (image_path.stem + '.txt')
            # scanpaths = pck.load(open(gaze_path, 'rb'), encoding='latin1')
            f = open(gaze_path, "r")
            index = 0
            _scanpaths = {}
            for row in f:
                if index == 0:  # pass the 1s line
                    index += 1
                    continue
                subject_id = row.split(' ')[0]
                lon = float(row.split(' ')[-2])
                # change to [bottom - top: 90 -> -90]
                lat = - float(row.split(' ')[-1])
                if not subject_id in _scanpaths:
                    _scanpaths[subject_id] = []
                    _scanpaths[subject_id].append([lat, lon])
                else:
                    _scanpaths[subject_id].append([lat, lon])
            temp = []
            for subject_id in _scanpaths:
                _scanpath = np.array(_scanpaths[subject_id])
                # if len(_scanpath) < 3:
                #     continue
                length_list.append(len(_scanpath))
                temp.append(torch.from_numpy(_scanpath))

            torch.save(temp, GAZE_PATH / (image_path.stem + '.pck'))
            origin_data = {"image_path": str(IMAGE_PATH / (str(image_path.stem) + '.jpg')),
                           "scanpaths": str(GAZE_PATH / (image_path.stem + '.pck'))}
            if phase != "train":
                data_dict[phase].append(origin_data)
            data_dict["all"].append(origin_data)

            for rotation_id in range(6):
                # f.write(str(image_path.stem) + '_' + str(rotation_id) + '.png' + "\n")
                rotation_angle = rotation_id * 60 - 180  # 【-180， 120】
                gaze_sphere = []
                for scanpath_id in range(0, len(temp)):
                    rotated_gaze = rotate_scanpath(temp[scanpath_id], rotation_angle)
                    gaze_sphere.append(rotated_gaze)

                torch.save(gaze_sphere, GAZE_PATH / (image_path.stem + '_' + str(rotation_id) + '.pck'))
                if phase == "train":
                    data = {"image_path": str(IMAGE_PATH / (str(image_path.stem) + '_' + str(rotation_id) + '.jpg')),
                            "scanpaths": str(GAZE_PATH / (image_path.stem + '_' + str(rotation_id) + '.pck'))}
                    data_dict[phase].append(data)

    tf = open(IMAGE_PATH.parent / "dataset.json", "w")
    json.dump(data_dict, tf)
    tf.close()

    print(f"mean length -- {mean(length_list)}")
