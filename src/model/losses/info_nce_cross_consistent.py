import torch
from torch import nn as nn
from torch.nn import functional as F
from torch.nn import KLDivLoss
import numpy as np

def listnet_loss(teacher_scores, student_scores, eps=1e-10):
    preds_smax = F.softmax(student_scores, dim=1)
    true_smax = F.softmax(teacher_scores, dim=1)
    preds_smax = preds_smax + eps
    preds_log = torch.log(preds_smax)
    cost = torch.mean(-torch.sum(true_smax * preds_log, dim=1))
    return cost

def kldiv_loss(teacher_scores, student_scores, eps=1e-10):
    preds_smax = F.log_softmax(student_scores, dim=1)
    true_smax = F.log_softmax(teacher_scores, dim=1)
    cost = F.kl_div(preds_smax, true_smax, log_target=True, reduction='batchmean')
    return cost

class InfoNCECrossConsistent(nn.Module):
    def __init__(self, 
                 dropout=0.1, 
                 single_modal_space_dim=256,
                 lambda_start_epoch=20,
                 lambda_end_epoch=50,
                 pose_out_dim=256, 
                 text_out_dim=256,
                 bkb_feats=False,
                 infonce_loss_fn=None,
                 temperature=0.1,
                 cross_consistent_type='listnet',   # 'kldiv',
                 use_length_as_pseudo_motion_label=False,
                 text_teacher_affects_m2m=True,
                 text_teacher_affects_t2t=True,
                 **kwargs):
        super().__init__()
        self.multimod_logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / temperature))
        self.m2m_logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / temperature))
        self.t2t_logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / temperature))
        self.t_teacher_logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / temperature))

        self.sentence_proj = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(text_out_dim, single_modal_space_dim)
        )

        self.pose_proj = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(pose_out_dim, single_modal_space_dim)
        )

        self.sim = lambda im, s: im.mm(s.t())
        # self.text_oracle_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        self.lambda_start_epoch = lambda_start_epoch
        self.lambda_end_epoch = lambda_end_epoch
        self.bkb_feats = bkb_feats
        self.infonce_loss_fn = infonce_loss_fn
        if cross_consistent_type == 'listnet':
            self.cc_loss = listnet_loss
        elif cross_consistent_type == 'kldiv':
            self.cc_loss = kldiv_loss

        self.use_length_as_pseudo_motion_label = use_length_as_pseudo_motion_label
        self.pseudo_mask_temperature = 10.0
        self.text_teacher_affects_m2m = text_teacher_affects_m2m
        self.text_teacher_affects_t2t = text_teacher_affects_t2t

    def motions_relevance_matrix_based_on_lengths(self, gt_lengths):
        rows = gt_lengths.unsqueeze(1).repeat(1, gt_lengths.size(0))
        cols = gt_lengths.unsqueeze(0).repeat(gt_lengths.size(0), 1)
        mat = torch.stack([rows, cols], dim=2)
        mat = torch.abs(mat[:, :, 0] - mat[:, :, 1])
        
        # the minus is to make the diagonal elements have the highest values
        return -mat

    def forward(self, im, s, sent_emb=None, gt_lengths=None, epoch=-1):
        assert epoch >= 0
        # im, s = comp_data['motion_emb'], comp_data['text_emb']
        im = torch.nn.functional.normalize(im, dim=-1)
        s = torch.nn.functional.normalize(s, dim=-1)
        
        # cosine similarity as logits
        logit_scale = self.multimod_logit_scale.exp()
        logits_per_image = logit_scale * im @ s.t()
        logits_per_text = logits_per_image.t()

        # produce single-modal features from im_bkb, d_bkb
        if self.bkb_feats:
            return NotImplementedError
            im_bkb, s_bkb = comp_data['motion_emb_bkb'], comp_data['text_emb_bkb']
            im_bkb = F.normalize(self.pose_proj(im_bkb), p=2, dim=1)
            s_bkb = F.normalize(self.sentence_proj(s_bkb), p=2, dim=1)
        else:
            im_bkb = im
            s_bkb = s

        # produce oracle similarities using sentence similarities
        # with torch.no_grad():
        #     texts = comp_data['texts']
        #     oracle_s_emb = self.text_oracle_model.encode(texts, convert_to_tensor=True, normalize_embeddings=True).to(im.device)

        # produce similarity matrices
        m2m_logits = self.m2m_logit_scale.exp() * im_bkb @ im_bkb.t()
        t2t_logits = self.t2t_logit_scale.exp() * s_bkb @ s_bkb.t()
        t_teacher_logits = self.t_teacher_logit_scale.exp() * sent_emb @ sent_emb.t()

        # listnet losses against teacher (text oracle weights)
        if self.text_teacher_affects_t2t:
            t2t_vs_teacher_loss = self.cc_loss(t_teacher_logits, t2t_logits)
        else:
            t2t_vs_teacher_loss = 0

        if self.text_teacher_affects_m2m:
            m2m_vs_teacher_loss = self.cc_loss(t_teacher_logits, m2m_logits)
        else:
            m2m_vs_teacher_loss = 0

        # listnet losses against learned multimodal embeddings
        # 1.symmetric listness loss for text
        multimod_embs_vs_t2t = (self.cc_loss(t2t_logits, logits_per_image) + self.cc_loss(t2t_logits, logits_per_text) + \
                                self.cc_loss(logits_per_image, t2t_logits) + self.cc_loss(logits_per_text, t2t_logits)) / 4
        # 2.symmetric listness loss for motion
        multimod_embs_vs_m2m = (self.cc_loss(m2m_logits, logits_per_image) + self.cc_loss(m2m_logits, logits_per_text) + \
                                self.cc_loss(logits_per_image, m2m_logits) + self.cc_loss(logits_per_text, m2m_logits)) / 4

        # compute bidirectional CE loss
        if self.infonce_loss_fn is None:
            num_logits = logits_per_image.shape[0]
            labels = torch.arange(num_logits, device=logits_per_image.device, dtype=torch.long)
            infonce_loss = (
                F.cross_entropy(logits_per_image, labels) +
                F.cross_entropy(logits_per_text, labels)
                ) / 2
        else:
            infonce_loss = self.infonce_loss_fn(im, s, sent_emb=sent_emb)

        # compute lambda as function of epoch
        # linear swipe between lambda_start_epoch and lambda_end_epoch
        lamb = (torch.Tensor([epoch]).to(im.device) - self.lambda_start_epoch) / (self.lambda_end_epoch - self.lambda_start_epoch)
        # clamp between 0 and 1
        lamb = lamb.clamp(0, 1)
        loss = infonce_loss + lamb * (multimod_embs_vs_m2m + multimod_embs_vs_t2t) + (1 - lamb) * (t2t_vs_teacher_loss + m2m_vs_teacher_loss)

        if self.use_length_as_pseudo_motion_label:
            # obtain how much the lengths in the batch (in every row?) are in line with the gt ones (in the diagonal)
            m2m_pseudo_mask = self.motions_relevance_matrix_based_on_lengths(gt_lengths)
            m2m_pseudo_mask /= self.pseudo_mask_temperature
            pseudo_loss = self.cc_loss(m2m_pseudo_mask, m2m_logits) + \
                          (self.cc_loss(m2m_pseudo_mask, logits_per_image) + self.cc_loss(m2m_pseudo_mask.t(), logits_per_text)) / 2
            
            loss += 0.2 * pseudo_loss   # FIXME: hardcoded weight. Should be a hyperparameter

        return loss
        
    def __repr__(self):
        return f"Contrastive: Cross-consistent (temp={self.temp})"