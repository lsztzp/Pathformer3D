import argparse
import os
from modules.transformer import Transformer
from mmcv import Config, DictAction
import cv2
import torch
import numpy as np
from torchvision.transforms import transforms
from metircs.suppor_lib import xyz2sphere

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a model or Inference')
    parser.add_argument('--config', default='config.py', help='config.py path')
    parser.add_argument('--device', default='cuda:3', help='cuda:n')
    parser.add_argument('--options', nargs='+', action=DictAction, help='arguments in dict')

    args = parser.parse_args()

    cfg = Config.fromfile(args.config)
    cfg.merge_from_dict(vars(args))

    if args.options is not None:
        cfg.merge_from_dict(args.options)

    model = Transformer(image_input_resize=cfg.image_input_resize, feature_dim=cfg.feature_dim,
                        patch_size=cfg.patch_size,
                        num_patch_h=cfg.num_patch_h,
                        num_patch_w=cfg.num_patch_w,
                        d_model=cfg.d_model, d_k=cfg.d_k, d_v=cfg.d_v, n_heads=cfg.n_heads, d_ff=cfg.d_ff,
                        dropout=cfg.dropout, enc_n_layers=cfg.enc_n_layers, postion_method=cfg.postion_method,
                        max_length=cfg.max_length, dec_n_layers=cfg.dec_n_layers,
                        MDN_hidden_num=cfg.MDN_hidden_num, num_gauss=cfg.num_gauss, action_map_size=cfg.action_map_size,
                        replace_encoder=cfg.replace_encoder
                        ).to(cfg.device)

    cfg.reload_path = "/data/lyt/02-Results/01-ScanPath/360_test_logs/01-05-baseline/checkpoint/checkpoint.pth.tar"

    assert cfg.reload_path
    checkpoint = torch.load(cfg.reload_path, map_location='cpu')
    model.load_state_dict(checkpoint['model'])

    input_1="/data/lyt/01-Datasets/01-ScanPath-Datasets/sitzmann/test_images/"
    input_2="/data/lyt/01-Datasets/01-ScanPath-Datasets/salient360!/test_images/"
    input_3 ="/data/qmengyu/01-Datasets/01-ScanPath-Dataset/AOI/Raw/TIP2021_AOI_Dataset/img_600/"
    input_4 ="/data/qmengyu/01-Datasets/01-ScanPath-Dataset/JUFE/images/"

    output_1 = "/data/lyt/02-Results/01-ScanPath/360_results_10paths/Ours/sitzmann/"
    output_2 = "/data/lyt/02-Results/01-ScanPath/360_results_10paths/Ours/salient360!/"
    output_3 = "/data/lyt/02-Results/01-ScanPath/360_results_10paths/Ours/aoi/"
    output_4 = "/data/lyt/02-Results/01-ScanPath/360_results_10paths/Ours/jufe/"

    path_in=[input_1,input_2,input_3,input_4]
    path_out=[output_1,output_2,output_3,output_4]

    for input_path,output_path in zip(path_in,path_out):
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        for filename in os.listdir(input_path):
            print(filename)
            path = input_path + filename

            img = cv2.imread(path, cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (cfg.image_input_resize[1], cfg.image_input_resize[0]),
                             interpolation=cv2.INTER_AREA)
            img = img.astype(np.float32) / 255.0
            transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
            img = transform(img)


            imgs = img.unsqueeze(dim=0).to(cfg.device)
            enc_masks = None
            batch_size = len(imgs)
            enc_inputs = model.feature_extrator(imgs)
            enc_outputs, enc_self_attns = model.encoder(enc_inputs, None)

            scanpaths = torch.zeros((10, 30, 2))
            for i in range(10):
                dec_inputs = torch.ones(batch_size, 1, 3).to(cfg.device) * 0.0
                for n in range(cfg.max_length):
                    dec_outputs, dec_self_attns, dec_enc_attns = model.decoder(enc_outputs, dec_inputs,
                                                                               enc_masks=enc_masks,
                                                                               dec_masks=torch.zeros(batch_size,
                                                                                                     n + 1).to(
                                                                                   cfg.device))

                    pis, mus, sigmas, rhos = model.mdn(dec_outputs)
                    outputs = model.mdn.sample_prob(pis, mus, sigmas, rhos).to(cfg.device)
                    last_fixations = outputs[:, -1]

                    dec_inputs = torch.cat((dec_inputs, last_fixations.unsqueeze(1)), dim=1)
                outputs = dec_inputs[:, 1:, :]
                output = outputs.squeeze()
                pred_fixations_sphere = xyz2sphere(output)
                scanpaths[i] = pred_fixations_sphere

            save_path = output_path + filename.split('.')[0] + '.pck'
            torch.save(scanpaths, save_path)