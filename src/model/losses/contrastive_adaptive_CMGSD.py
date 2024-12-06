import torch
from torch import nn as nn
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer

class ContrastiveAdaptiveCMGSD(nn.Module):
    """
    Compute contrastive loss with adaptive margin, as proposed in 
    Improving Video Retrieval by Adaptive Margin (SIGIR '21) https://dl.acm.org/doi/pdf/10.1145/3404835.3462927
    """

    def __init__(self, 
        margin=0.2, 
        dropout=0.1, 
        single_modal_space_dim=256,
        lambda_start_epoch=20,
        lambda_end_epoch=50,
        pose_out_dim=256, 
        text_out_dim=256,
        max_violation_after=0):

        super().__init__()
        self.sim = lambda im, s: im.mm(s.t())
        self.pose_proj = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(pose_out_dim, single_modal_space_dim)
        )
        self.sentence_proj = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(text_out_dim, single_modal_space_dim)
        )
        self.text_oracle_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

        self.mi = 0.2
        self.beta = 1
        self.alpha = margin

        self.margin = margin
        self.lambda_start_epoch = lambda_start_epoch
        self.lambda_end_epoch = lambda_end_epoch
        self.max_violation_after = max_violation_after

    @staticmethod
    def triplet_loss_caption_retrieval(scores, margin):
        diagonal = scores.diag().view(scores.size(0), 1)
        d1 = diagonal.expand_as(scores)

        # compare every diagonal score to scores in its column
        cost = (margin + scores - d1).clamp(min=0)

        # clear diagonals
        mask = torch.eye(scores.size(0)) > .5
        I = mask
        if torch.cuda.is_available():
            I = I.cuda()
        cost = cost.masked_fill_(I, 0)

        return cost

    @staticmethod
    def triplet_loss_motion_retrieval(scores, margin):
        diagonal = scores.diag().view(scores.size(0), 1)
        d1 = diagonal.t().expand_as(scores)

        # compare every diagonal score to scores in its column
        cost = (margin + scores - d1).clamp(min=0)

        # clear diagonals
        mask = torch.eye(scores.size(0)) > .5
        I = mask
        if torch.cuda.is_available():
            I = I.cuda()
        cost = cost.masked_fill_(I, 0)

        return cost

    def forward(self, comp_data, epoch=0, return_similarity_mat=False):
        # the cross-modal features
        im, s = comp_data['motion_emb'], comp_data['text_emb']
        scores = self.sim(im, s)

        # produce oracle similarities using sentence similarities
        with torch.no_grad():
            texts = comp_data['texts']
            oracle_s_emb = self.text_oracle_model.encode(texts, convert_to_tensor=True, normalize_embeddings=True).to(im.device)

        # produce single-modal features from im_bkb, d_bkb
        im_bkb, s_bkb = comp_data['motion_emb_bkb'], comp_data['text_emb_bkb']
        im_bkb = F.normalize(self.pose_proj(im_bkb), p=2, dim=1)
        s_bkb = F.normalize(self.sentence_proj(s_bkb), p=2, dim=1)

        # omega and tau functions
        omega_dyn = 1 - self.sim(im_bkb, im_bkb)    # dynamic omega
        tau_dyn = 1 - self.sim(s_bkb, s_bkb)        # dynamic tau
        tau_static = 1 - self.sim(oracle_s_emb, oracle_s_emb)   # static tau

        # remapping
        m_omega_dyn = ((omega_dyn - omega_dyn.mean()) / omega_dyn.std()) * self.beta + self.alpha
        m_tau_dyn = ((tau_dyn - tau_dyn.mean()) / tau_dyn.std()) * self.beta + self.alpha
        m_tau_static = ((tau_static - tau_static.mean()) / tau_static.std()) * self.beta + self.alpha

        # negative margins are dangerous, I think
        m_omega_dyn = m_omega_dyn.clamp(min=0)
        m_tau_dyn = m_tau_dyn.clamp(min=0)
        m_tau_static = m_tau_static.clamp(min=0)

        # compute lambda as function of epoch
        # linear swipe between lambda_start_epoch and lambda_end_epoch
        # if epoch == self.lambda_start_epoch:
        #     print('here')
        lamb = (torch.Tensor([epoch]).to(im.device) - self.lambda_start_epoch) / (self.lambda_end_epoch - self.lambda_start_epoch)
        # clamp between 0 and 1
        lamb = lamb.clamp(0, 1)

        max_violation = epoch >= self.max_violation_after

        # losses
        cap_retrieval_loss = self.triplet_loss_caption_retrieval(scores, margin=self.margin) + \
            lamb * (self.triplet_loss_caption_retrieval(scores, margin=m_omega_dyn) + self.triplet_loss_caption_retrieval(scores, margin=m_tau_dyn)) + \
            (1 - lamb) * (self.triplet_loss_caption_retrieval(scores, margin=m_tau_static))  # we don't have m_omega_static
        if max_violation:
            cap_retrieval_loss = cap_retrieval_loss.max(1)[0]
        cap_retrieval_loss = cap_retrieval_loss.mean()

        mot_retrieval_loss = self.triplet_loss_motion_retrieval(scores, margin=self.margin) + \
            lamb * (self.triplet_loss_motion_retrieval(scores, margin=m_omega_dyn) + self.triplet_loss_motion_retrieval(scores, margin=m_tau_dyn)) + \
            (1 - lamb) * (self.triplet_loss_motion_retrieval(scores, margin=m_tau_static))  # we don't have m_omega_static
        if max_violation:
            mot_retrieval_loss = mot_retrieval_loss.max(0)[0]
        mot_retrieval_loss = mot_retrieval_loss.mean()

        loss = cap_retrieval_loss + mot_retrieval_loss

        monitors = {
            'm_omega_dyn_mean': m_omega_dyn.flatten().mean().item(),
            'm_omega_dyn_std': m_omega_dyn.flatten().std().item(),
            'm_tau_dyn_mean': m_tau_dyn.flatten().mean().item(),
            'm_tau_dyn_std': m_tau_dyn.flatten().std().item(),
            'm_tau_static_mean': m_tau_static.flatten().mean().item(),
            'm_tau_static_std': m_tau_static.flatten().std().item(),
            'lambda': lamb
        }
        return loss, monitors