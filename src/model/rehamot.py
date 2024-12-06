import torch
import torch.backends.cudnn as cudnn
import torch.nn.init
import torch.nn as nn
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch.nn.utils.clip_grad import clip_grad_norm_
from einops import reduce
from pytorch_lightning import LightningModule
from typing import List, Dict, Optional
from torch import Tensor

from retrieval import compute_sim_matrix
from .metrics import all_contrastive_metrics

def mask_logits(inputs, mask, mask_value=-1e30):
    mask = mask.type(torch.float32)
    return inputs + (1.0 - mask) * mask_value

class CosineSim():
    def __init__(self):
        pass

    def __call__(self, mm, s, *args):
        """Cosine similarity between all the motion and sentence pairs
        """
        return mm.mm(s.t())

class CrossPerceptualSalienceMapping():
    def __init__(self):
        pass

    def __call__(self, motion, query, c_mask, q_mask):
        # (batch_size, m_seq_len, q_seq_len)
        score = torch.einsum('amd,bqd->abmq', motion, query)
        # query-wise softmax (m_batch_size, q_batch_size, m_seq_len, q_seq_len)
        score_m = nn.Softmax(dim=3)(mask_logits(score, q_mask.unsqueeze(0).unsqueeze(2)))
        # motion-wise softmax (m_batch_size, q_batch_size, m_seq_len, q_seq_len)
        score_t = nn.Softmax(dim=2)(mask_logits(score, c_mask.unsqueeze(1).unsqueeze(-1)))
        # (m_batch_size, q_batch_size, q_seq_len, m_seq_len)
        score_t = score_t.transpose(2, 3) 
        # m2t perceptual similarity
        score_m = reduce(score * score_m, 'a b m q -> a b m', 'sum')
        # m2t salience similarity
        score_m = reduce(score_m, 'a b m -> a b', 'max')
        # t2m perceptual similarity
        score_t = reduce(score.transpose(2, 3) * score_t, 'a b q m -> a b q', 'sum')
        # t2m salience similarity (q_batch_size, m_batch_size)
        score_t = reduce(score_t, 'a b q -> a b', 'max')
        score = 1/2 * score_m + 1/2 * score_t
        return score

def adjust_learning_rate(lr_update, optimizer, epoch):
    """Sets the learning rate to the initial LR
       decayed by 10 at [lr_update] epoch"""
    lr_multiplier = (0.1 ** (1 if (epoch // lr_update) > 0 else 0))
    for param_group in optimizer.param_groups:
        param_group['lr'] *= lr_multiplier

class Rehamot(LightningModule):
    """
    Rehamot: coss-modal retrieval human motion and text
    """

    def __init__(self,
                 text_encoder,
                 motion_encoder,
                 contrastive_loss,
                 sim,
                 lr: float,
                 # grad_clip: float,
                 # device: str,
                 finetune: bool,
                 enable_momentum: bool,
                 warm_up: int,
                 lr_update: int,
                 threshold_selfsim_metrics: float,
                 lmd: Dict = {"contrastive": 1.0},
                 **kwargs):
        super().__init__()

        self.warm_up = warm_up
        self.lr_update = lr_update
        self.finetune = finetune
        self.learning_rate = lr
        self.enable_momentum = enable_momentum
        self.textencoder = text_encoder
        self.motionencoder = motion_encoder
        self.threshold_selfsim_metrics = threshold_selfsim_metrics
        self.lmd = lmd
        # self.grad_clip = grad_clip
        # if sim == "cpsmapping":
        #     self.sim = cross_perceptual_salience_mapping
        #     only_return_cls_token = False
        # elif sim == "cosine":
        #     self.sim = cosine_sim
        #     only_return_cls_token = True
        # else:
        #     raise NotImplementedError()
        # Build Models
        # self.motionencoder = instantiate(
        #     motion_encoder, only_return_cls_token=only_return_cls_token)
        # self.textencoder = instantiate(text_encoder, only_return_cls_token=only_return_cls_token)
        # if torch.cuda.is_available():
        #     torch.backends.cudnn.enabled = True
        num_params = sum(p.numel() for p in self.motionencoder.parameters() if p.requires_grad)
        print(f"Number of parameters in Rehamot's motionencoder: {num_params}")
        num_params = sum(p.numel() for p in self.textencoder.parameters() if p.requires_grad)
        print(f"Number of parameters in Rehamot's textencoder: {num_params}")

        # Loss and Optimizer
        self.criterion = contrastive_loss #instantiate(contrastive_loss, sim=self.sim)
        self.criterion.sim = sim    # bad way to inject this...

        self.use_hard_negative_mining = False
        self.lr_updated = False
        self.validation_step_t_latents = []
        self.validation_step_m_latents = []
        self.validation_step_sent_emb = []

    def configure_optimizers(self):
        params = []
        # Fine-tuning with different learning rates for parts of the neural network
        if self.finetune:
            lr_multiplier = 10
            for prefix, module in [('motion', self.motionencoder), ('text', self.textencoder)]:
                for name, param in module.named_parameters():
                    lr = self.learning_rate * lr_multiplier if any(name.startswith(
                        s) for s in module.learning_rates_x) else self.learning_rate
                    params.append({'name': name, 'params': param, 'lr': lr})
        else:
            params += [{'name': name, 'params': param, 'lr': self.learning_rate}
                       for name, param in self.motionencoder.named_parameters()]
            params += [{'name': name, 'params': param, 'lr': self.learning_rate}
                       for name, param in self.textencoder.named_parameters()]

        self.params = [param['params'] for param in params]

        return {"optimizer": torch.optim.Adam(params)}
    
    def training_step(self, batch: Dict, batch_idx: int) -> Tensor:
        if batch_idx == 0:
            # at the beginning of each epoch
            if self.current_epoch >= self.warm_up and not self.use_hard_negative_mining:
                self.hard_negative_mining()
                self.use_hard_negative_mining = True
                # print.info('use hard negative mining')

            if self.current_epoch >= self.lr_update and not self.lr_updated:
                adjust_learning_rate(self.lr_update, self.optimizers(), self.current_epoch)
                self.lr_updated = True

        bs = len(batch["motion_x_dict"]["x"])
        losses = self.compute_loss(batch)

        for loss_name in sorted(losses):
            loss_val = losses[loss_name]
            self.log(
                f"train_{loss_name}",
                loss_val,
                on_epoch=True,
                on_step=True,
                batch_size=bs,
            )
        return losses["loss"]

    def validation_step(self, batch: Dict, batch_idx: int) -> Tensor:
        bs = len(batch["motion_x_dict"]["x"])
        losses, t_latents, m_latents = self.compute_loss(batch, return_all=True)

        # Store the latent vectors
        self.validation_step_t_latents.append(t_latents)
        self.validation_step_m_latents.append(m_latents)
        self.validation_step_sent_emb.append(batch["sent_emb"])

        for loss_name in sorted(losses):
            loss_val = losses[loss_name]
            self.log(
                f"val_{loss_name}",
                loss_val,
                on_epoch=True,
                on_step=True,
                batch_size=bs,
            )

        return losses["loss"]

    def on_validation_epoch_end(self):
        # abs_errors = torch.cat(self.length_regression_absolute_errors)
        # variances = torch.cat(self.length_regression_variances)

        dataset = self.trainer.val_dataloaders.dataset
        self.eval()
        res = compute_sim_matrix(
            self, dataset, dataset.keyids, batch_size=64
        )
        contrastive_metrics = all_contrastive_metrics(
            res['sim_matrix'],
            emb=res["sent_emb"],
            threshold=self.threshold_selfsim_metrics,
        )

        # for recall validation metrics, ignore recall@1, recall@2, given that they are quite unstable
        contrastive_metrics['rsum'] = sum([v for k, v in contrastive_metrics.items() if '/R' in k]) #and int(k.split('/R')[-1]) >= 3])
        contrastive_metrics['rsum-t2m'] = sum([v for k, v in contrastive_metrics.items() if 't2m/R' in k]) #and int(k.split('/R')[-1]) >= 3])
        contrastive_metrics['medr'] = sum([v for k, v in contrastive_metrics.items() if '/MedR' in k])

        for loss_name in sorted(contrastive_metrics):
            loss_val = contrastive_metrics[loss_name]
            self.log(
                f"val_{loss_name}_epoch",
                loss_val,
                on_epoch=True,
                on_step=False,
            )

        # self.log(f"val_motion_length_abs_error", abs_errors.mean(), on_epoch=True, on_step=False)
        # self.log(f"val_motion_length_variance", variances.mean(), on_epoch=True, on_step=False)

        self.validation_step_t_latents.clear()
        self.validation_step_m_latents.clear()
        self.validation_step_sent_emb.clear()

        self.train()
    
    def compute_loss(self, batch: Dict, return_all: bool = False) -> Dict:
        text_x_dict = batch["text_x_dict"]
        motion_x_dict = batch["motion_x_dict"]

        motion_emb, text_emb, motion_mask, text_mask = self.forward_emb(motion_x_dict, text_x_dict)
        loss = self.forward_loss(motion_emb, text_emb, motion_mask, text_mask)
        losses = {"contrastive": loss}

        losses["loss"] = sum(
            self.lmd[x] * val for x, val in losses.items() if x in self.lmd
        )

        if return_all:
            return losses, text_emb, motion_emb

        return losses

    # def state_dict(self):
    #     state_dict = [self.motionencoder.state_dict(),
    #                   self.textencoder.state_dict()]
    #     return state_dict

    # def load_state_dict(self, state_dict):
    #     self.motionencoder.load_state_dict(state_dict[0])
    #     self.textencoder.load_state_dict(state_dict[1])

    # def train_start(self):
    #     """switch to train mode
    #     """
    #     self.motionencoder.train()
    #     self.textencoder.train()

    # def val_start(self):
    #     """switch to evaluate mode
    #     """
    #     self.motionencoder.eval()
    #     self.textencoder.eval()

    def hard_negative_mining(self, flag=True):
        self.criterion.max_violation = flag

    def _find_encoder(self, inputs, modality):
        assert modality in ["text", "motion", "auto"]

        if modality == "text" or "text" in inputs:
            return self.textencoder
        elif modality == "motion":
            return self.motionencoder

        m_nfeats = self.motionencoder.nfeats
        t_nfeats = self.textencoder.nfeats

        if m_nfeats == t_nfeats:
            raise ValueError(
                "Cannot automatically find the encoder, as they share the same input space."
            )

        nfeats = inputs["x"].shape[-1]
        if nfeats == m_nfeats:
            return self.motionencoder
        elif nfeats == t_nfeats:
            return self.textencoder
        else:
            raise ValueError("The inputs is not recognized.")

    def encode(
        self,
        inputs,
        modality: str = "auto",
        sample_mean: Optional[bool] = False,
    ):
        # Encode the inputs
        encoder = self._find_encoder(inputs, modality)
        encoded = encoder(inputs)
        encoded = encoded[0] if isinstance(encoded, tuple) else encoded # take the actual feature, discarding the mask

        return encoded

    def forward_emb(self, motion, text, **kwargs):
        """Compute the motion and text embeddings
        """
        motion_emb, motion_mask = self.motionencoder(motion)
        text_emb, text_mask = self.textencoder(text)
        return motion_emb, text_emb, motion_mask, text_mask

    def forward_loss(self, motion_emb, text_emb, motion_mask=None, text_mask=None, idx=None, is_train=True):
        """Compute the loss given pairs of motion and text embeddings
        """
        n = motion_emb[0].size(0)
        loss, drop_num = self.criterion(motion_emb, text_emb, motion_mask, text_mask)
        # self.logger.update('Le', loss.item(), n)
        # self.logger.update('Drop_num', drop_num, n)
        return loss

    # def train_emb(self, motion, text, length, index, **kwargs):
    #     """One training step given motions and texts.
    #     """
    #     self.Eiters += 1
    #     self.logger.update('Eit', self.Eiters)
    #     self.logger.update('lr', self.optimizer.param_groups[0]['lr'])

    #     # compute the embeddings
    #     motion_emb, text_emb, motion_mask, text_mask = self.forward_emb(motion, text, length)

    #     # measure similarity in a mini-batch
    #     if self.Eiters % kwargs['val_step'] == 0 or kwargs['init']:
    #         self.log_similarity(motion_emb, text_emb, motion_mask, text_mask)

    #     # measure accuracy and record loss
    #     self.optimizer.zero_grad()
    #     loss = self.forward_loss(motion_emb, text_emb, motion_mask, text_mask, index)

    #     # compute gradient and do SGD step
    #     loss.backward()
    #     if self.grad_clip > 0:
    #         clip_grad_norm_(self.params, self.grad_clip)
    #     self.optimizer.step()
    
    def get_similarity(self, motion_emb, text_emb, motion_mask=None, text_mask=None):
        return self.criterion.sim(motion_emb, text_emb, motion_mask, text_mask)

    # def log_similarity(self, motion_emb, text_emb, motion_mask, text_mask):
    #     """Measure similarity in a mini-batch.
    #     """
    #     # compute similarity matrix
    #     # the key is connect with LogCollector
    #     similarity_matrices = {
    #         'sim_matrix_inter': self.sim(motion_emb, text_emb, motion_mask, text_mask).detach().cpu().numpy(),
    #         'sim_matrix_m': self.sim(motion_emb, motion_emb, motion_mask, motion_mask).detach().cpu().numpy(),
    #         'sim_matrix_t': self.sim(text_emb, text_emb, text_mask, text_mask).detach().cpu().numpy()
    #     }
    #     # add similarity matrix and mean similarity to tensorboard
    #     for name, similarity_matrix in similarity_matrices.items():
    #         fig = plot_similarity_matrix(similarity_matrix, name)
    #         self.logger.tb_figure(name, fig, self.Eiters)