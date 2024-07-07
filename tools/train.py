import torch
from modules.mdn3d import mixture_probability
from utils.utils import build_scheduler, save_checkpoint


class Trainer:
    def __init__(self, lr, dataloder, work_dir, device, start_epoch, epoch_nums, val_step, writer):
        self.lr = lr
        self.dataloder = dataloder
        self.work_dir = work_dir
        self.device = device
        self.cur_epoch = start_epoch
        self.epoch_nums = epoch_nums
        self.val_step = val_step
        self.writer = writer

    def train_epochs(self, model, optimizer, lr_scheduler, evaluation, sphere_constraint_loss):
        if lr_scheduler:
            scheduler = build_scheduler(lr_scheduler=lr_scheduler, optimizer=optimizer)

        best_epoch, best_score = 0, 99999
        for self.cur_epoch in range(self.cur_epoch, self.epoch_nums + 1):
            model.train()
            train_performance, batch_count = {'loss': 0}, 0
            for i_batch, batch in enumerate(self.dataloder):
                optimizer.zero_grad()

                train_batch_size = batch['imgs'].shape[0]
                max_length = batch['dec_inputs'].shape[-2]

                pis, mus, sigmas, rhos = model(batch['imgs'].to(self.device),
                                               batch['dec_inputs'].float().to(self.device),
                                               batch['dec_masks'].to(self.device))
                # 混合密度网络损失
                probs = mixture_probability(pis, mus, sigmas, rhos,
                                            batch['scanpath'].unsqueeze(-1).to(self.device)).squeeze()

                probs_mask = torch.arange(max_length).expand(train_batch_size, max_length). \
                    lt(batch['valid_len'].unsqueeze(-1).expand(train_batch_size, max_length)).to(
                    self.device)  # 筛选合法注视点预测的概率

                probs = torch.masked_select(probs, probs_mask)
                # print(f"probs-----max: {probs.max()}---min: {probs.min()} ---sum: {probs.sum()} ---all: {probs}")

                # test sphere_constraint_loss
                if sphere_constraint_loss:
                    probs_sphere = mixture_probability(pis, mus, sigmas, rhos,
                                                       batch['sphere_coordinates'].unsqueeze(1).to(
                                                           self.device)).squeeze()
                    loss = torch.mean(-torch.log(probs)) + 0.1 * torch.mean(-torch.log(probs_sphere))
                else:
                    loss = torch.mean(-torch.log(probs))

                batch_count += 1
                loss_item = loss.detach().cpu().item()
                train_performance['loss'] += loss_item

                print(f'train_(epoch{self.cur_epoch:3d}/{i_batch:4d}), ' + f'loss: {loss_item:.3f}')
                loss.backward()
                optimizer.step()


            train_performance['loss'] /= batch_count
            self.writer.add_scalar('AA_Scalar/train_loss', train_performance['loss'], self.cur_epoch)
            self.writer.add_scalar('AA_Scalar/train_lr', float(optimizer.param_groups[0]['lr']), self.cur_epoch)

            if lr_scheduler:
                scheduler.step()

            if (self.cur_epoch + 1) % self.val_step == 0:
                save_checkpoint(self.cur_epoch, model, optimizer, self.work_dir)
                score = evaluation.validation(model, self.cur_epoch, dataset_name='sitzmann', save=False)
                if score < best_score:
                    best_score = score
                    best_epoch = self.cur_epoch
                evaluation.validation(model, self.cur_epoch, dataset_name='salient360', save=False)
                evaluation.validation(model, self.cur_epoch, dataset_name='aoi', save=False)
                evaluation.validation(model, self.cur_epoch, dataset_name='jufe', save=False)

        return best_epoch
