from collections import defaultdict
from statistics import mean
import numpy as np
from pathlib import Path
import torch
from PIL import Image
from metircs.metrics_scanpath import levenshtein_distance, DTW, REC, DET, TDE, scan_match
from metircs.suppor_lib import sphere2plane

dataset_path = Path("/data/qmengyu/01-Datasets/01-ScanPath-Dataset/")
dataset_dict = {
    "sitzmann": {
        "image_size": [128, 256],
        "data_path": Path("/data/qmengyu/01-Datasets/01-ScanPath-Dataset/sitzmann/fixations/"),
        "image_path": Path("/data/qmengyu/01-Datasets/01-ScanPath-Dataset/sitzmann/images/"),
        "suffix": ".png",
        "regexp": "cubemap_[0-9][0-9][0-9][0-9].pck"
    },

    "salient360": {
        "image_size": [128, 256],
        "data_path": Path("/data/qmengyu/01-Datasets/01-ScanPath-Dataset/salient360/fixations/"),
        "image_path": Path("/data/qmengyu/01-Datasets/01-ScanPath-Dataset/salient360/images/"),
        "suffix": ".jpg",
        "regexp": "*[0-9][0-9].pck"
    },

    "aoi": {
        "image_size": [128, 256],
        "data_path": Path("/data/qmengyu/01-Datasets/01-ScanPath-Dataset/AOI/fixations/"),
        "image_path": Path("/data/qmengyu/01-Datasets/01-ScanPath-Dataset/AOI/images/"),
        "suffix": ".jpg",
        "regexp": "*[A-Z0-9]?.pck"
    },

    "jufe": {
        "image_size": [128, 256],
        "data_path": Path("/data/qmengyu/01-Datasets/01-ScanPath-Dataset/JUFE/fixations/"),
        "image_path": Path("/data/qmengyu/01-Datasets/01-ScanPath-Dataset/JUFE/images/"),
        "suffix": ".png",
        "regexp": "*.pck"
    }
}


def score_seq(pre, gt, dataset_name, metrics=('LEV', 'DTW', 'REC', 'DET', 'ScanMatch', 'TDE')):
    """
    pre 预测扫视路径， gt 真实扫视路径
    """
    resize_image_size = dataset_dict[dataset_name.lower()]["image_size"]

    threshold = 2 * 6
    scores = {}
    if 'LEV' in metrics:
        scores['d-LEV'] = levenshtein_distance(pre, gt, resize_image_size[0], resize_image_size[1])  # 高宽？？
    if 'DTW' in metrics:
        scores['d-DTW'] = DTW(pre, gt)
    if 'REC' in metrics:
        score_REC = REC(pre, gt, threshold)
        if np.isnan(score_REC):
            scores['u-REC'] = 0.  # 设置阈值
        else:
            scores['u-REC'] = score_REC  # 设置阈值
    if 'DET' in metrics:
        score_DET = DET(pre, gt, threshold)
        if np.isnan(score_DET):
            scores['u-DET'] = 0.  # 设置阈值
        else:
            scores['u-DET'] = score_DET  # 设置阈值
    if 'TDE' in metrics:
        scores['d-TDE'] = TDE(pre, gt)
    if 'ScanMatch' in metrics:
        scores["u-ScanMatch"] = scan_match(pre, gt, height=resize_image_size[0], width=resize_image_size[1])

    return scores


def score_all_gts(pred, gts, dataset_name, metrics=('LEV', 'DTW', 'REC', 'DET', 'ScanMatch', 'TDE')):
    """
    pre 预测扫视路径, 图片形式  【 L, 2】
    gts 多个真实扫视路径 图片形式  【 N, L, 2】
    """
    scores_all_gts = defaultdict(list)
    for n in range(len(gts)):
        gt = gts[n].astype(int)
        scores = score_seq(pred.astype(int), gt, dataset_name, metrics)

        for metric, score in scores.items():
            scores_all_gts[metric].append(score)

    for metric, scores_list in scores_all_gts.items():
        scores_list = [score for score in scores_list if score > 1e-4]
        if scores_list:
            scores_all_gts[metric] = mean(scores_list)
        else:
            scores_all_gts[metric] = 0.

    return scores_all_gts


# 待验证
def get_score_filename(pred_fixation_sphere, file_name, dataset_name,
                       metrics=('LEV', 'DTW', 'REC', 'DET', 'ScanMatch', 'TDE')):
    """
    pred_fixation 预测扫视路径 经纬度形式： 【 L, 2】 np 形式
    file_name 真值文件名
    """
    data_path = dataset_dict[dataset_name.lower()]["data_path"]
    imagespath = dataset_dict[dataset_name.lower()]["image_path"]
    resize_image_size = dataset_dict[dataset_name.lower()]["image_size"]
    suffix = dataset_dict[dataset_name.lower()]["suffix"]

    image_path = imagespath / (file_name + suffix)
    image = Image.open(image_path)
    image_height_width = [image.height, image.width]

    gt_fixations_sphere = torch.load(data_path / (file_name + ".pck"))

    gt_fixations_plane = []
    for index in range(len(gt_fixations_sphere)):
        gt_fixations_plane.append(sphere2plane(gt_fixations_sphere[index], image_height_width).numpy())
    # gt_fixations_plane = np.array(gt_fixations_plane)
    for index in range(len(gt_fixations_plane)):
        gt_fixations_plane[index][:, 0] *= resize_image_size[0] / image.height
        gt_fixations_plane[index][:, 1] *= resize_image_size[1] / image.width

    pred_fixations_plane = sphere2plane(torch.from_numpy(pred_fixation_sphere), image_height_width)

    pred_fixations_plane[:, 0] *= resize_image_size[0] / image.height
    pred_fixations_plane[:, 1] *= resize_image_size[1] / image.width
    pred_fixations_plane = pred_fixations_plane.numpy()

    clamp_lens = {
        "sitzmann": 30,
        "salient360": 25,
        "aoi": 8,
        "jufe": 15
    }

    clamp_len = clamp_lens[dataset_name]
    temp = np.zeros((clamp_len, 2))
    current_length = len(pred_fixations_plane)
    # print(f"均步采样 {current_length} to {clamp_len}")
    step = int(current_length / clamp_len)
    for sample_i in range(clamp_len):
        temp[sample_i] = pred_fixations_plane[sample_i * step]
    pred_fixations_plane = temp

    return score_all_gts(pred_fixations_plane, gt_fixations_plane, dataset_name, metrics)


# 待验证
def get_score_file(pred_dir_path, dataset_name, metrics=('LEV', 'DTW', 'REC', 'DET', 'ScanMatch', 'TDE')):
    """
    pred_file_path 预测扫视路径文件路径
    """
    dataset_name = dataset_name.lower()
    scores_all_data = defaultdict(list)

    pred_dir_path = Path(pred_dir_path) / dataset_name
    for file_path in pred_dir_path.glob("*"):
        print(file_path)
        pred_fixations_sphere = torch.load(file_path).numpy()

        if pred_fixations_sphere.shape == (10,30,2):
            for i in range(len(pred_fixations_sphere)):
                pred_fixations_sphere_one = pred_fixations_sphere[i]
                scores_all_gts = get_score_filename(pred_fixations_sphere_one, file_path.stem, dataset_name, metrics=metrics)
                for metric, score in scores_all_gts.items():
                    scores_all_data[metric].append(score)
        else:
            scores_all_gts = get_score_filename(pred_fixations_sphere, file_path.stem, dataset_name, metrics=metrics)

            for metric, score in scores_all_gts.items():
                scores_all_data[metric].append(score)

    for metric, scores_list in scores_all_data.items():
        scores_list = [score for score in scores_list if score > 1e-4]
        if scores_list:
            scores_all_data[metric] = mean(scores_list)
        else:
            scores_all_data[metric] = 0.

    return scores_all_data
