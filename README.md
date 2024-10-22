# Pathformer3D: A 3D Scanpath Transformer for 360° Images"

You can find the preprint on [[arXiv](https://arxiv.org/abs/2407.10563)]

Scanpath prediction in 360° images can help realize rapid rendering and better user interaction in Virtual/Augmented Reality applications. However, existing scanpath prediction models for 360° images execute scanpath prediction on 2D equirectangular projection plane, which always result in big computation error owing to the 2D plane's distortion and coordinate discontinuity. In this work, we perform scanpath prediction for 360° images in 3D spherical coordinate system and proposed a novel 3D scanpath Transformer named Pathformer3D. Specifically, a 3D Transformer encoder is first used to extract 3D contextual feature representation for the 360° image. Then, the contextual feature representation and historical fixation information are input into a Transformer decoder to output current time step's fixation embedding, where the self-attention module is used to imitate the visual working memory mechanism of human visual system and directly model the time dependencies among the fixations. Finally, a 3D Gaussian distribution is learned from each fixation embedding, from which the fixation position can be sampled. Evaluation on four panoramic eye-tracking datasets demonstrates that Pathformer3D outperforms the current state-of-the-art methods.
# Installation

```bash
conda create -n  Pahtformer3D python=3.9
conda activate Pathformer3D
bash install.sh
```

# Scripts
## Trainning
1. To reproduce the training and validation dataset, please referring to [dataloder.py](./datasets/dataloder.py) for placing your dataset files.
2. Execute

``` python3 main.py --work_dir="baseline" --device="cuda:0" --config="config.py" ```
## Test
For evaluation, run the following process to generate scanpaths and metrics scores:

```bash
python3 inference.py
python3 compare.py 
 ```
Acknowledgement - A portion of our code is adapted from the [repository](https://github.com/xiangjieSui/ScanDMM) for [ScanDMM](https://openaccess.thecvf.com/content/CVPR2023/papers/Sui_ScanDMM_A_Deep_Markov_Model_of_Scanpath_Prediction_for_360deg_CVPR_2023_paper.pdf) model. We would like to thank the authors Sui et al. for open-sourcing their code. 
# Bibtex 
If you find the code useful in your research, please consider citing the paper.
```
  @inproceedings{quan2025pathformer3d,
    title={Pathformer3D: A 3D Scanpath Transformer for 360° Images},
    author={Quan, Rong and Lai, Yantao and Qiu, Mengyu and Liang, Dong},
    booktitle={European Conference on Computer Vision},
    year={2025}
}
```


