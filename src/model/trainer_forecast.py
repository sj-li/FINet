import datetime
from pathlib import Path
import time
import pickle
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import MetricCollection
from torch.optim.lr_scheduler import CosineAnnealingLR
from src.metrics import MR, minADE, minFDE, brier_minFDE
from src.utils.optim import WarmupCosLR
from src.utils.submission_av2 import SubmissionAv2
from src.utils.LaplaceNLLLoss import LaplaceNLLLoss
from .model_forecast import ModelForecast

import os
import matplotlib.pyplot as plt
import numpy as np

from typing import List

model_dict = {
    'ModelForecast': ModelForecast,  # only 'FINet'
}


class Trainer(pl.LightningModule):
    def __init__(
        self,
        model: dict,
        pretrained_weights: str = None,
        lr: float = 1e-3,
        warmup_epochs: int = 10,
        epochs: int = 60,
        weight_decay: float = 1e-4,
        ws_offset: List = [0.3, 1.0],
    ) -> None:
        super(Trainer, self).__init__()
        self.warmup_epochs = warmup_epochs
        self.epochs = epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.save_hyperparameters()
        self.submission_handler = SubmissionAv2()

        self.model_type = model.pop('type')
        assert self.model_type in model_dict
        self.net = model_dict[self.model_type](**model)

        # self.net = self.get_model(model_type)(**model)

        if pretrained_weights is not None:
            self.net.load_from_checkpoint(pretrained_weights)
            print('Pretrained weights have been loaded.')

        metrics = MetricCollection(
            {
                "minADE1": minADE(k=1),
                "minADE6": minADE(k=6),
                "minFDE1": minFDE(k=1),
                "minFDE6": minFDE(k=6),
                "MR": MR(),
                "b-minFDE6": brier_minFDE(k=6),
            }
        )
        self.laplace_loss = LaplaceNLLLoss()
        self.val_metrics = metrics.clone(prefix="val_")
        self.val_metrics_new = metrics.clone(prefix="val_new_")
        
        self.ws_offset = ws_offset

        self.total_time=0
        self.cur_time = 0

        self.count = np.zeros(6)
        self.count_closet = np.zeros(6)
        
    

    def forward(self, data):
        return self.net(data)

    def predict(self, data):
        memory_dict = None
        predictions = []
        probs = []
        for i in range(len(data)):
            cur_data = data[i]
            cur_data['memory_dict'] = memory_dict
            out = self(cur_data)
            memory_dict = out['memory_dict']
            prediction, prob = self.submission_handler.format_data(
                cur_data, out["y_hat"], out["pi"], inference=True)
            predictions.append(prediction)
            probs.append(prob)

        return predictions, probs

    def cal_loss(self, out, data, tag=''):
        y_hat, pi, y_hat_others = out["y_hat"], out["pi"], out["y_hat_others"]
        scal, scal_new = out["scal"], out["scal_new"]
        new_y_hat = out.get("new_y_hat", None)
        new_pi = out.get("new_pi", None)
        dense_predict = out.get("dense_predict", None)
        ep_offsets = out.get("ep_offsets", None)
        
        center = data["x_centers"][:,0]

        # gt
        y, y_others = data["target"][:, 0], data["target"][:, 1:]

        # loss for output of state query
        if dense_predict is not None:
            if isinstance(dense_predict, list):
                dense_reg_loss = 0
                for pred in dense_predict:
                    dense_reg_loss = dense_reg_loss + F.smooth_l1_loss(pred, y)
            else:
                dense_reg_loss = F.smooth_l1_loss(dense_predict, y)
        else:
            dense_reg_loss = 0
        
        
        if ep_offsets is not None:
            gt_offsets = y[:,-1] - center
            if isinstance(ep_offsets, list):
                ep_reg_loss = 0
                for w, pred in zip(self.ws_offset, ep_offsets):
                    # ep_reg_loss = dense_reg_loss + w*F.smooth_l1_loss(pred, gt_offsets)
                    ep_reg_loss = ep_reg_loss + w*F.smooth_l1_loss(pred, gt_offsets)
            else:
                ep_reg_loss = F.smooth_l1_loss(ep_offsets, gt_offsets)
        else:
            ep_reg_loss = 0
        # gt_offsets = y[:,-1] - center
        # ep_reg_loss = dense_reg_loss + F.smooth_l1_loss(ep_offsets[-1], gt_offsets)
        

        if y_hat.dim() == 3:
            # loss for output of mode query
            ep = y[:,-1,:2]
            l2_norm = torch.norm(y_hat[..., :2] - ep.unsqueeze(1), dim=-1)
            best_mode = torch.argmin(l2_norm, dim=-1)
            y_hat_best = y_hat[torch.arange(y_hat.shape[0]), best_mode]
            agent_reg_loss = F.smooth_l1_loss(y_hat_best[..., :2], ep)
            agent_cls_loss = F.cross_entropy(pi, best_mode.detach(), label_smoothing=0.2)
        else:   
            # loss for output of mode query
            l2_norm = torch.norm(y_hat[..., :2] - y.unsqueeze(1), dim=-1).sum(dim=-1)
            best_mode = torch.argmin(l2_norm, dim=-1)
            y_hat_best = y_hat[torch.arange(y_hat.shape[0]), best_mode]
            agent_reg_loss = F.smooth_l1_loss(y_hat_best[..., :2], y)
            agent_cls_loss = F.cross_entropy(pi, best_mode.detach(), label_smoothing=0.2)
        
        # loss for final output
        if new_y_hat is not None:
            l2_norm_new = torch.norm(new_y_hat[..., :2] - y.unsqueeze(1), dim=-1).sum(dim=-1)
            best_mode_new = torch.argmin(l2_norm_new, dim=-1)
            new_y_hat_best = new_y_hat[torch.arange(new_y_hat.shape[0]), best_mode_new]
            new_agent_reg_loss = F.smooth_l1_loss(new_y_hat_best[..., :2], y)
        else:
            new_agent_reg_loss = 0
        if new_pi is not None:
            new_pi_reg_loss = F.cross_entropy(new_pi, best_mode_new.detach(), label_smoothing=0.2)
        else:
            new_pi_reg_loss = 0

        # loss for other agents
        others_reg_mask = data["target_mask"][:, 1:]
        others_reg_loss = F.smooth_l1_loss(
            y_hat_others[others_reg_mask], y_others[others_reg_mask]
        )

        # Laplace loss, which is not necessary
        predictions = {}
        predictions['traj'] = y_hat
        predictions['scale'] = scal
        predictions['probs'] = pi
        laplace_loss = self.laplace_loss.compute(predictions, y) 

        predictions['traj'] = new_y_hat
        predictions['scale'] = scal_new
        predictions['probs'] = new_pi
        laplace_loss_new = self.laplace_loss.compute(predictions, y)
        

                
        loss = new_agent_reg_loss + new_pi_reg_loss + laplace_loss_new
        loss = loss + agent_reg_loss + agent_cls_loss + laplace_loss + others_reg_loss + dense_reg_loss
        loss = loss + ep_reg_loss
                


        disp_dict = {
            f"{tag}loss": loss.item(),
            f"{tag}reg_loss": agent_reg_loss.item(),
            f"{tag}cls_loss": agent_cls_loss.item(),
            f"{tag}others_reg_loss": others_reg_loss.item(),
            f"{tag}laplace_loss": laplace_loss.item(),
            f"{tag}laplace_loss_new": laplace_loss_new.item(),
        }
        
        if new_y_hat is not None:
            disp_dict[f"{tag}reg_loss_refine"] = new_agent_reg_loss.item()
        if new_pi is not None:
            disp_dict[f"{tag}reg_loss_new_pi"] = new_pi_reg_loss.item()
        if dense_predict is not None:
            disp_dict[f"{tag}reg_loss_dense"] = dense_reg_loss.item()
        if ep_offsets is not None:
            disp_dict[f"{tag}ep_reg_loss"] = ep_reg_loss.item()

        return loss, disp_dict

    def training_step(self, data, batch_idx):
        if isinstance(data, list):
            data = data[-1]
            
        
        out = self(data)
        loss, loss_dict = self.cal_loss(out, data)

        for k, v in loss_dict.items():
            self.log(
                f"train/{k}",
                v,
                on_step=True,
                on_epoch=True,
                prog_bar=False,
                sync_dist=True,
            )

        return loss

    def vis_ep(self, ep, gt, prediction, save_path="/home/shijie/code/FINet_ICCV2025/FINet/Visual_ep"):
        prediction = prediction.cpu().detach().numpy()
        gt = gt.cpu().detach().numpy()
        ep1 = ep[0].cpu().detach().numpy()[0]
        ep2 = ep[1].cpu().detach().numpy()[0]



        fig, ax = plt.subplots(figsize=(6, 6))
        
        for pred in prediction:
            ax.plot(pred[:, 0], pred[:, 1], color='blue', linewidth=2)
        
        ax.plot(gt[:, 0], gt[:, 1], color='red', linewidth=2)

        plt.scatter(ep1[0], ep1[1], color='green', label='Point (5, 10)', zorder=5)  # Plot the point
        plt.text(ep1[0], ep1[1], f'(Endpoint 1)', fontsize=12, ha='left', va='bottom')  # Optionally add a label

        plt.scatter(ep2[0], ep2[1], color='green', label='Point (5, 10)', zorder=5)  # Plot the point
        plt.text(ep2[0], ep2[1], f'(Endpoint 2)', fontsize=12, ha='left', va='bottom')  # Optionally add a label

        # ax.set_xlim(0, 400)
        # ax.set_ylim(0, 400)
        plt.tight_layout()

        
        self.cur_time = self.cur_time + 1
        save_path = os.path.join(save_path, str(self.cur_time) + ".png")
        print(save_path)
        plt.savefig(save_path, dpi=300, pad_inches=0)

        
    def validation_step(self, data, batch_idx):
        if isinstance(data, list):
            data = data[-1]
        try:
            out = self(data)
            _, loss_dict = self.cal_loss(out, data)
            metrics = self.val_metrics(out, data['target'][:, 0])
            if out['new_y_hat'] is not None:
                out['y_hat'] = out['new_y_hat']
                out['pi'] = out['new_pi']
            if out['new_y_hat'] is not None:
                metrics_new = self.val_metrics_new(out, data['target'][:, 0])

            # print(out['ep_offsets'], data["target"][:,0,0], data["target"][:,0,-1])

            # self.vis_ep(out['ep_offsets'], data["target"][0,0], out['new_y_hat'][0])

            # ep = out['new_y_hat'][0,:,-1]
            # ep_gt = data["target"][0,0,-1]


            # self.count[out['new_pi'].argmax().item()]  += 1
            # self.count_closet[((ep - ep_gt)**2).sum(-1).argmin().item()] += 1
        
            # print(self.count)
    
            self.log_dict(
                metrics,
                prog_bar=True,
                on_step=False,
                on_epoch=True,
                batch_size=1,
                sync_dist=True,
            )
            if out['new_y_hat'] is not None:
                self.log_dict(
                    metrics_new,
                    prog_bar=True,
                    on_step=False,
                    on_epoch=True,
                    batch_size=1,
                    sync_dist=True,
                )

        except:
            pass

        # print(self.count, self.count_closet)

    def on_test_start(self) -> None:
        save_dir = Path("./submission")
        save_dir.mkdir(exist_ok=True)
        self.submission_handler = SubmissionAv2(
            save_dir=save_dir
        )
    
    def on_test_end(self) -> None:

       latency = self.total_time / len(self.test_dataloader().dataset)

    def test_step(self, data, batch_idx) -> None:
        if isinstance(data, list):
            data = data[-1]
        
        torch.cuda.synchronize()
        start_time = time.time()

        out = self(data)

        torch.cuda.synchronize()
        end_time = time.time()
        
        latency = end_time - start_time
        self.total_time += latency

        self.cur_time += 1

        # print(self.total_time/self.cur_time)
        
        if out['new_y_hat'] is not None:
            out['y_hat'] = out['new_y_hat']
            out['pi'] = out['new_pi']
        self.submission_handler.format_data(data, out["y_hat"], out["pi"])

    def on_test_end(self) -> None:
        self.submission_handler.generate_submission_file()

    def configure_optimizers(self):
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (
            nn.Linear,
            nn.Conv1d,
            nn.Conv2d,
            nn.Conv3d,
            nn.MultiheadAttention,
            nn.LSTM,
            nn.GRU,
        )
        blacklist_weight_modules = (
            nn.BatchNorm1d,
            nn.BatchNorm2d,
            nn.BatchNorm3d,
            nn.SyncBatchNorm,
            nn.LayerNorm,
            nn.Embedding,
        )
        for module_name, module in self.named_modules():
            for param_name, param in module.named_parameters():
                full_param_name = (
                    "%s.%s" % (module_name, param_name) if module_name else param_name
                )
                if "bias" in param_name:
                    no_decay.add(full_param_name)
                elif "weight" in param_name:
                    if isinstance(module, whitelist_weight_modules):
                        decay.add(full_param_name)
                    elif isinstance(module, blacklist_weight_modules):
                        no_decay.add(full_param_name)
                elif not ("weight" in param_name or "bias" in param_name):
                    no_decay.add(full_param_name)
        param_dict = {
            param_name: param for param_name, param in self.named_parameters()
        }
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0

        optim_groups = [
            {
                "params": [
                    param_dict[param_name] for param_name in sorted(list(decay))
                ],
                "weight_decay": self.weight_decay,
            },
            {
                "params": [
                    param_dict[param_name] for param_name in sorted(list(no_decay))
                ],
                "weight_decay": 0.0,
            },
        ]

        optimizer = torch.optim.AdamW(
            optim_groups, lr=self.lr, weight_decay=self.weight_decay
        )
        scheduler = WarmupCosLR(
            optimizer=optimizer,
            lr=self.lr,
            min_lr=1e-5,
            warmup_epochs=self.warmup_epochs,
            epochs=self.epochs,
        )
        return [optimizer], [scheduler]

