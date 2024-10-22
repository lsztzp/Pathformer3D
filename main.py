import argparse
import os
from torch import optim
from datasets.dataloder import Scanpath360Dataloder
from tools.evaluation import Evaluation
from torch.utils.tensorboard import SummaryWriter
from modules.transformer import Transformer
from tools.train import Trainer
from mmcv import Config, DictAction
from datetime import datetime
from utils.utils import setup_seed, loadCheckpoint

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a model or Inference')
    parser.add_argument('--config', default='config.py', help='config.py path')
    parser.add_argument('--work_dir', default=argparse.SUPPRESS, help='path to save logs and weights')
    parser.add_argument('--device', default='cuda:0', help='cuda:n')
    parser.add_argument('--wo_train', action="store_true", help='w/o train the model')
    parser.add_argument('--wo_inference', action="store_true", help='w/o inference to scanpath results')
    parser.add_argument('--wo_score', action="store_true", help='w/o score scanpath results')
    parser.add_argument('--options', nargs='+', action=DictAction, help='arguments in dict')

    args = parser.parse_args()

    cfg = Config.fromfile(args.config)
    cfg.merge_from_dict(vars(args))

    if args.options is not None:
        cfg.merge_from_dict(args.options)

    assert cfg.work_dir
    if not cfg.wo_train:
        cfg.work_dir = os.path.join('/data/lyt/02-Results/01-ScanPath/logs/',
                                    datetime.today().strftime('%m-%d-') + cfg.work_dir)
    else:
        assert cfg.reload_path
        cfg.work_dir = os.path.join('/data/lyt/02-Results/01-ScanPath/logs/', cfg.work_dir)
    if not cfg.wo_train:
        writer = SummaryWriter(log_dir=cfg.work_dir)
    else:
        writer = None

    setup_seed(cfg.seed)

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

    if not cfg.feature_grad:
        for p in model.feature_extrator.parameters():
            p.requires_grad = False

    optimizer = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    if cfg.reload_path:  # finetune or score: load checkpiont from file reload_path
        print(cfg.reload_path)
        epoch_start, model, optimizer = loadCheckpoint(model=model, optimizer=optimizer, checkpointPath=cfg.reload_path)
    else:  # resume from break-point: reload checkpoint from dir workdir
        epoch_start, model, optimizer = loadCheckpoint(model=model, optimizer=optimizer, work_dir=cfg.work_dir)

    print('-----------------model results will be in :', cfg.work_dir)
    cfg.dump(os.path.join(cfg.work_dir, 'config.py'))

    evaluation = Evaluation(work_dir=cfg.work_dir, writer=writer,
                            val_batch_size=cfg.val_batch_size,
                            device=cfg.device, seed=cfg.seed, max_length=cfg.max_length,
                            action_map_size=cfg.action_map_size, image_input_resize=cfg.image_input_resize,
                            patch_size=cfg.patch_size, )

    if not cfg.wo_train:
        train_dataloder = Scanpath360Dataloder(dataset_name=cfg.train_dataset, phase='train',
                                               batch_size=cfg.train_batch_size,
                                               image_input_resize=cfg.image_input_resize,
                                               patch_size=cfg.patch_size, max_length=cfg.max_length, seed=cfg.seed)

        train = Trainer(lr=cfg.lr, dataloder=train_dataloder, work_dir=cfg.work_dir, device=cfg.device,
                        start_epoch=epoch_start, epoch_nums=cfg.epoch_nums, val_step=cfg.val_step,
                        writer=writer)

        best_epoch = train.train_epochs(model, optimizer, lr_scheduler=cfg.lr_scheduler,
                                        evaluation=evaluation, sphere_constraint_loss=cfg.sphere_constraint_loss)

        # load the best performance checkpoint
        print(f"----------- best epoch is - {best_epoch} ")
        epoch_start, model, optimizer = loadCheckpoint(model=model, optimizer=optimizer, epoch=best_epoch,
                                                       work_dir=cfg.work_dir)
        score_prefix = str(best_epoch)
    else:
        best_epoch = epoch_start
        score_prefix = cfg.reload_path.split('/')[-1].split('.')[0]

    if not cfg.wo_inference:
        # inference results
        evaluation.validation(model, best_epoch, dataset_name='sitzmann', save=True, )
        evaluation.validation(model, best_epoch, dataset_name='jufe', save=True)
        evaluation.validation(model, best_epoch, dataset_name='salient360', save=True)
        evaluation.validation(model, best_epoch, dataset_name='aoi', save=True)
