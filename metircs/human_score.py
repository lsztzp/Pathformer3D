from collections import defaultdict
from statistics import mean
import numpy as np
import torch
from PIL import Image

from metircs.utils import score_all_gts, dataset_dict
from metircs.suppor_lib import sphere2plane

sitzmann_test_set = ['cubemap_0000.pck', 'cubemap_0006.pck', 'cubemap_0009.pck']

def score_human(source, metrics, ):
    print(f'computing {source} dataset in {metrics}')
    gtspath = dataset_dict[source.lower()]["data_path"]
    imagespath = dataset_dict[source.lower()]["image_path"]
    regexp = dataset_dict[source.lower()]["regexp"]
    suffix = dataset_dict[source.lower()]["suffix"]
    resize_image_size = dataset_dict[source.lower()]["image_size"]

    scores_all_data = defaultdict(list)
    for file_path in gtspath.glob(regexp):
        if source == 'sitzmann' and file_path.name not in sitzmann_test_set:
            continue
        image_path = imagespath / (file_path.stem + suffix)
        image = Image.open(image_path)
        image_height_width = [image.height, image.width]

        gt_fixations_sphere = torch.load(file_path)
        gt_fixations_plane = []
        for index in range(len(gt_fixations_sphere)):
            gt_fixations_plane.append(sphere2plane(gt_fixations_sphere[index], image_height_width).numpy())
        gt_fixations_plane = np.array(gt_fixations_plane)
        for index in range(len(gt_fixations_plane)):
            gt_fixations_plane[index][:, 0] *= resize_image_size[0] / image.height
            gt_fixations_plane[index][:, 1] *= resize_image_size[1] / image.width

        print(file_path, "  ", len(gt_fixations_plane))

        scores_all_human = defaultdict(list)
        for n in range(len(gt_fixations_plane)):
            scanpath = gt_fixations_plane[n].astype(np.float)
            other = np.delete(gt_fixations_plane, n, axis=0)
            scores_gts = score_all_gts(scanpath, other, dataset_name=source, metrics=metrics)

            # add this human score
            for metric, score in scores_gts.items():
                scores_all_human[metric].append(score)

        # process all human scores as this data score
        for metric, scores_list in scores_all_human.items():
            scores_list = [score for score in scores_list if score > 1e-4]
            if scores_list:
                scores_all_data[metric].append(mean(scores_list))
            else:
                scores_all_data[metric].append(0.)

    # prcess all data scores
    for metric, scores_list in scores_all_data.items():
        scores_list = [score for score in scores_list if score > 1e-4]
        if scores_list:
            scores_all_data[metric] = mean(scores_list)
        else:
            scores_all_data[metric] = 0.
    return scores_all_data


if __name__ == "__main__":

    sources = ('sitzmann', "salient360!", "aoi", "jufe")
    # sources = ("aoi", )
    metrics = ('LEV', 'DTW', 'REC', 'DET', 'ScanMatch', 'TDE')

    results = []
    with open("./human_score.txt", 'w') as f:
        for source in sources:
            result = score_human(source, metrics)
            f.write(f"\n --------- {source} ----------- \n")
            for metric, score in result.items():
                f.write(f"{metric}   :      {score} \n")
            results.append(score_human(source, metrics))

    print(results)
