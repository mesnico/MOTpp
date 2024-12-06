import torch
import torch.nn.functional as F


# For reference
# https://stats.stackexchange.com/questions/7440/kl-divergence-between-two-univariate-gaussians
# https://pytorch.org/docs/stable/_modules/torch/distributions/kl.html#kl_divergence
class KLLoss:
    def __call__(self, q, p):
        mu_q, logvar_q = q
        mu_p, logvar_p = p

        log_var_ratio = logvar_q - logvar_p
        t1 = (mu_p - mu_q).pow(2) / logvar_p.exp()
        div = 0.5 * (log_var_ratio.exp() + t1 - 1 - log_var_ratio)
        return div.mean()

    def __repr__(self):
        return "KLLoss()"
    

def kldiv_loss(teacher_scores, student_scores, eps=1e-10):
    preds_smax = F.log_softmax(student_scores, dim=1)
    true_smax = F.log_softmax(teacher_scores, dim=1)
    cost = F.kl_div(preds_smax, true_smax, log_target=True, reduction='batchmean')
    return cost


class InfoNCE_with_filtering:
    def __init__(self, temperature=0.7, threshold_selfsim=0.8, use_length_as_pseudo_motion_label=False):
        self.temperature = temperature
        self.threshold_selfsim = threshold_selfsim
        self.use_length_as_pseudo_motion_label = use_length_as_pseudo_motion_label
        self.pseudo_mask_temperature = 10.0

    def get_sim_matrix(self, x, y):
        x_logits = torch.nn.functional.normalize(x, dim=-1)
        y_logits = torch.nn.functional.normalize(y, dim=-1)
        sim_matrix = x_logits @ y_logits.T
        return sim_matrix
    
    def motions_relevance_matrix_based_on_lengths(self, gt_lengths):
        rows = gt_lengths.unsqueeze(1).repeat(1, gt_lengths.size(0))
        cols = gt_lengths.unsqueeze(0).repeat(gt_lengths.size(0), 1)
        mat = torch.stack([rows, cols], dim=2)
        mat = torch.abs(mat[:, :, 0] - mat[:, :, 1])
        
        # the minus is to make the diagonal elements have the highest values
        return -mat

    def __call__(self, x, y, sent_emb=None, epoch=None, gt_lengths=None):
        bs, device = len(x), x.device
        sim_matrix = self.get_sim_matrix(x, y) / self.temperature
        sim_matrix_orig = sim_matrix.clone()

        if sent_emb is not None and self.threshold_selfsim:
            # put the threshold value between -1 and 1
            real_threshold_selfsim = 2 * self.threshold_selfsim - 1
            # Filtering too close values
            # mask them by putting -inf in the sim_matrix
            selfsim = sent_emb @ sent_emb.T
            selfsim_nodiag = selfsim - selfsim.diag().diag()
            idx = torch.where(selfsim_nodiag > real_threshold_selfsim)
            sim_matrix[idx] = -torch.inf

        labels = torch.arange(bs, device=device)

        total_loss = (
            F.cross_entropy(sim_matrix, labels) + F.cross_entropy(sim_matrix.T, labels)
        ) / 2

        if self.use_length_as_pseudo_motion_label:
            # obtain how much the lengths in the batch (in every row?) are in line with the gt ones (in the diagonal)
            m2m_pseudo_mask = self.motions_relevance_matrix_based_on_lengths(gt_lengths)
            m2m_pseudo_mask /= self.pseudo_mask_temperature
            pseudo_loss = (kldiv_loss(m2m_pseudo_mask, sim_matrix_orig) + kldiv_loss(m2m_pseudo_mask.t(), sim_matrix_orig.t())) / 2
            
            total_loss += 0.2 * pseudo_loss

        return total_loss

    def __repr__(self):
        return f"Contrastive: InfoNCE with filtering (temp={self.temp})"
