import json

import numpy as np
import torch
from torch import nn
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms
from metircs.suppor_lib import sphere2xyz
from pathlib import Path
import cv2

default_datasets_dir = Path("/data/qmengyu/01-Datasets/01-ScanPath-Dataset")
paths = {"sitzmann_data_dict_path": str(default_datasets_dir / 'sitzmann/dataset.json'),
         "salient360_data_dict_path": str(default_datasets_dir / 'salient360/dataset.json'),
         "aoi_data_dict_path": str(default_datasets_dir / 'AOI/dataset.json'),
         "jufe_data_dict_path": str(default_datasets_dir / 'JUFE/dataset.json'),

         "merge_data_dict_path": "/data/lyt/01-Datasets/01-ScanPath-Datasets/dataset.json"
         }

class ScanPath360Dataset(Dataset):
    """

    """

    def __init__(self, phase, dataset_name, image_input_resize, patch_size=(16, 16), max_length=18):
        self.phase = phase
        self.dataset_name = dataset_name.lower()
        self.image_input_resize = image_input_resize

        self.max_length = max_length
        self.pool2d = nn.AvgPool2d(patch_size)
        self.transform = torchvision.transforms.Compose([
            transforms.ToTensor(),
        ])
        with open(paths[self.dataset_name + "_data_dict_path"], 'r', encoding='utf-8') as load_f:
            self.data_list_dict = json.load(load_f)[self.phase]

        if self.phase == "train":  # preload data
            print(f"--------------- {self.dataset_name} ---{self.phase} data loading")
            image_list = {}
            for data in self.data_list_dict:
                if data["image_path"] not in image_list:

                    path = data["image_path"]
                    img = cv2.imread(path, cv2.IMREAD_COLOR)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img_size = torch.tensor([img.shape[:2]])

                    img = cv2.resize(img, (self.image_input_resize[1], self.image_input_resize[0]),
                                     interpolation=cv2.INTER_AREA)
                    img = img.astype(np.float32) / 255.0
                    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
                    img = transform(img)

                    image_list[data["image_path"]] = {
                        "img": img,
                        "img_size": img_size
                    }
                data["img"] = image_list[data["image_path"]]['img']
                data["img_size"] = image_list[data["image_path"]]['img_size']

                scanpath_index = 0
                if self.phase == "train":
                    scanpath_index = data["index"]
                gt_fixations_sphere = torch.load(data["scanpaths"])[scanpath_index]
                gt_fixations_xyz = sphere2xyz(gt_fixations_sphere)
                data["gt_fixations_xyz"] = gt_fixations_xyz

    def __getitem__(self, index):
        data = self.data_list_dict[index]
        img_name = data["image_path"].split('/')[-1]
        if self.phase == "train":
            img = data['img']
            img_size = data['img_size']
            gt_fixations_xyz = data["gt_fixations_xyz"]
        else:
            path = data["image_path"]
            img = cv2.imread(path, cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_size = torch.tensor([img.shape[:2]])

            img = cv2.resize(img, (self.image_input_resize[1], self.image_input_resize[0]),
                             interpolation=cv2.INTER_AREA)
            img = img.astype(np.float32) / 255.0
            transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
            img = transform(img)


            scanpath_index = 0
            if self.phase == "train":
                scanpath_index = data["index"]
            gt_fixations_sphere = torch.load(data["scanpaths"])[scanpath_index]
            gt_fixations_xyz = sphere2xyz(gt_fixations_sphere)

        scanpath = torch.zeros(self.max_length, 3)  # [L, 3]
        valid_len = len(gt_fixations_xyz)

        if valid_len <= self.max_length:
            scanpath[:valid_len] = gt_fixations_xyz
        else:
            scanpath = gt_fixations_xyz[:self.max_length]

        # dec_input
        dec_input = torch.zeros(self.max_length, 3)
        dec_input[1:] = scanpath[:-1]
        dec_input[0, :] = 0
        # dec_input[0, :] = 0.5

        # dec_mask
        dec_mask = torch.zeros(self.max_length)
        dec_mask[valid_len:] = 1

        sphere_coordinates = torch.randn(3, 10000)
        sphere_coordinates /= sphere_coordinates.norm(2, dim=0)

        return {
            'imgs': img,
            'scanpath': scanpath,
            'valid_len': valid_len,
            'dec_inputs': dec_input,
            'dec_masks': dec_mask,
            'file_names': img_name[:-4],
            'img_sizes': img_size,
            'sphere_coordinates': sphere_coordinates
        }

    def __len__(self):
        return len(self.data_list_dict)


class Scanpath360Dataloder(DataLoader):
    """

    """

    def __init__(self, dataset_name, phase, batch_size,
                 image_input_resize, patch_size, max_length, seed=1218):
        self.seed = seed
        self.dataset = ScanPath360Dataset(phase=phase, dataset_name=dataset_name,
                                          image_input_resize=image_input_resize, patch_size=patch_size,
                                          max_length=max_length)

        super().__init__(dataset=self.dataset, batch_size=batch_size, shuffle=True,
                         num_workers=16, drop_last=phase == "train", pin_memory=True,
                         worker_init_fn=self._init_fn)

    def _init_fn(self, worker_id):
        np.random.seed(int(self.seed) + worker_id)
