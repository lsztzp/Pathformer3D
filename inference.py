import argparse
import os
from modules.transformer import Transformer
from mmcv import Config, DictAction
import cv2
import torch
import numpy as np
from torchvision.transforms import transforms
from metircs.suppor_lib import plot_scanpaths,xyz2plane

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a model or Inference')
    parser.add_argument('--config', default='config.py', help='config.py path')
    parser.add_argument('--device', default='cuda:2', help='cuda:n')
    parser.add_argument('--options', nargs='+', action=DictAction, help='arguments in dict')

    args = parser.parse_args()

    cfg = Config.fromfile(args.config)
    cfg.merge_from_dict(vars(args))

    if args.options is not None:
        cfg.merge_from_dict(args.options)

    cfg.reload_path = "/data/lyt/02-Results/01-ScanPath/360_logs/01-05-num_gauss=5/checkpoint/checkpoint.pth.tar"
    input_path = "/data/lyt/02-Results/01-ScanPath/inference_input/"
    output_path = "/data/lyt/02-Results/01-ScanPath/inference_output/"

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

    assert cfg.reload_path
    checkpoint = torch.load(cfg.reload_path, map_location='cpu')
    model.load_state_dict(checkpoint['model'])

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    for filename in os.listdir(input_path):
        for i in range(10):
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
            dec_inputs = torch.ones(batch_size, 1, 3).to(cfg.device) * 0.0
            enc_inputs = model.feature_extrator(imgs)
            enc_outputs, enc_self_attns = model.encoder(enc_inputs, None)
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
            pred_fixations_plane = xyz2plane(output).cpu().numpy()

            image_path = input_path + filename
            save_path = output_path + str(i)+"_"+ filename

            plot_scanpaths(pred_fixations_plane, img_path=str(image_path), save_path=str(save_path),
                           img_height=512,
                           img_witdth=1024)





