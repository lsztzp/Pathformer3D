from pathlib import Path
from metircs.utils import get_score_file

def results_to_scores(dir_path, prefix="ep", sources=('sitzmann', "salient360!", "aoi", "jufe"),
                      metrics=('LEV', 'DTW', 'REC', 'DET', 'ScanMatch', 'TDE')):
    dir_path = Path(dir_path)

    with open(dir_path / f"{prefix}_scores.txt", "w") as f:
        for source in sources:
            result = get_score_file(str(dir_path), source, metrics=metrics)
            f.write(f"\n --------- {source} ----------- \n")
            for metric, score in result.items():
                f.write(f"{metric}   :      {score} \n")
    print(f" -------------- scores result save at {dir_path}/ep_scores.txt")

if __name__ == "__main__":
    # Given the result saving path, calculate all index scores of all data sets and save them in the path under score.txt
    # compare_results_dir = Path("/data/lyt/02-Results/01-ScanPath/360_results_10paths/")
    # for method_path in compare_results_dir.glob("*"):
    #     results_to_scores(str(method_path))

    pred_dir_path = Path("/data/lyt/02-Results/01-ScanPath/360_results_10paths/Ours/")
    results_to_scores(str(pred_dir_path), sources=('sitzmann', "salient360", "aoi", "jufe"))