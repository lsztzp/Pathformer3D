import copy
import json
import os
import random
import xlrd
from pathlib import Path
import numpy as np
import torch

rotate = False  # 是否执行原图旋转到指定目录

IMAGE_ORIGIN_PATH = Path("/data/qmengyu/01-Datasets/01-ScanPath-Dataset/JUFE/Raw/JUFE/dis/")
IMAGE_PATH = Path("/data/qmengyu/01-Datasets/01-ScanPath-Dataset/JUFE/images/")

GAZE_ORIGIN_PATH = Path("/data/qmengyu/01-Datasets/01-ScanPath-Dataset/JUFE/Raw/JUFE/HMData_revised_final/")
MOS_PATH = Path("/data/qmengyu/01-Datasets/01-ScanPath-Dataset/JUFE/Raw/JUFE/mos.xls")
GAZE_PATH = Path("/data/qmengyu/01-Datasets/01-ScanPath-Dataset/JUFE/fixations/")

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
                      output_path + name.split('.')[0] + '_' + str(i) + '.png'
                os.system(cmd)


def search_dir(root, target_file, re):
    for root_dir, dirs, files in os.walk(root):
        for file in files:
            if file == target_file:
                file_path = os.path.join(root_dir, file)
                re.append(file_path)
    return re


def getFileName(root, target_file):
    re = []
    res_list = search_dir(root, target_file, re)
    return res_list


if __name__ == '__main__':

    if rotate:  # 图像旋转
        rotate_images(str(IMAGE_PATH) + '/', str(IMAGE_PATH) + '/')

    MOS_file = xlrd.open_workbook(MOS_PATH).sheet_by_name('Sheet1')
    image_list = MOS_file.col_values(1)[1:]
    index = 0
    for img in image_list:
        image_list[index] = img.split('.')[0]
        index += 1
    index = 0

    arr = list(range(1, 259))
    random.shuffle(arr)
    cnt = int(len(arr) * 0.8)
    # cnt = 0         # 暂时没有训练集，全部用于验证
    train_set = arr[:cnt]
    test_set = arr[cnt:]
    train_set.sort()
    test_set.sort()

    data_dict = {
        "all": [],  # 当数据集整体用于测试时使用
        "train": [],
        "test": []
    }

    cnt = 0
    for file_name in os.listdir(IMAGE_ORIGIN_PATH):
        image_name = file_name.split('.')[0]
        its_index = image_name.split('_')[0]
        # phase = ('train', 'test')[its_index in self.test_set]
        if int(its_index) in test_set:
            phase = 'test'
        else:
            phase = 'train'

        csv_path = getFileName(GAZE_ORIGIN_PATH, image_name + '.csv')
        scanpaths = []
        for csv in csv_path:
            data = xlrd.open_workbook(csv).sheet_by_name('Sheet1')
            lat, lon = np.array(data.col_values(
                3)).reshape(-1, 1), np.array(data.col_values(4)).reshape(-1, 1)
            _gaze = np.concatenate((lat, lon), axis=1)
            gaze = []
            # FPS = 1, sample 1 points per second
            fps = 1
            num_sample = fps * 15
            step = int(_gaze.shape[0] / num_sample)
            for j in range(num_sample):
                gaze.append(_gaze[j * step])
            scanpaths.append(gaze)

        image_path = Path(str(IMAGE_PATH) + '/' + file_name)

        scanpaths = np.array(scanpaths)
        torch.save(torch.from_numpy(scanpaths), GAZE_PATH / (image_path.stem + '.pck'))
        origin_data = {
            "image_path": str(IMAGE_PATH / (image_name + '.png')),
            "scanpaths": str(GAZE_PATH / (image_name + '.pck'))
        }
        data_dict["all"].append(origin_data)

        if phase == "train":
            for scanpath_id in range(0, scanpaths.shape[0]):
                data_index = {
                    "image_path": str(IMAGE_PATH / (image_name + '.png')),
                    "scanpaths": str(GAZE_PATH / (image_name + '.pck')),
                    "index": scanpath_id
                }
                data_dict[phase].append(data_index)
        else:
            data_dict[phase].append(origin_data)

        print("%d / %d " % (cnt, 258 * 4))
        cnt += 1

    tf = open(IMAGE_PATH.parent / "dataset.json", "w")
    json.dump(data_dict, tf)
    tf.close()
