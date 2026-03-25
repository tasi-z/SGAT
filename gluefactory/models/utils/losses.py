import torch
import torch.nn as nn
from omegaconf import OmegaConf
from gluefactory.geometry.epipolar import generalized_epi_dist, relative_pose_error

# def batch_episym(x1, x2, F):
#     batch_size, num_pts = x1.shape[0], x1.shape[1]
#     x1 = torch.cat([x1, x1.new_ones(batch_size, num_pts,1)], dim=-1).reshape(batch_size, num_pts,3,1)
#     x2 = torch.cat([x2, x2.new_ones(batch_size, num_pts,1)], dim=-1).reshape(batch_size, num_pts,3,1)
#     F = F.reshape(-1,1,3,3).repeat(1,num_pts,1,1)
#     x2Fx1 = torch.matmul(x2.transpose(2,3), torch.matmul(F, x1)).reshape(batch_size,num_pts)
#     Fx1 = torch.matmul(F,x1).reshape(batch_size,num_pts,3)
#     Ftx2 = torch.matmul(F.transpose(2,3),x2).reshape(batch_size,num_pts,3)
#     ys = (x2Fx1**2 * (
#             1.0 / (Fx1[:, :, 0]**2 + Fx1[:, :, 1]**2 + 1e-15) +
#             1.0 / (Ftx2[:, :, 0]**2 + Ftx2[:, :, 1]**2 + 1e-15))).sqrt()
#     return ys
    
def CELoss(seed_x0,seed_x1,data,confidence,inlier_th,batch_mask=1):
    #seed_x: b*k*2
    # ys=batch_episym(seed_x1,seed_x2,e)
    ys=n_epi_err = generalized_epi_dist(
        seed_x0[None],
        seed_x1[None],
        data["view0"]["camera"],
        data["view1"]["camera"],
        data["T_0to1"],
        False,
        essential=True,
    )[0]
    mask_pos,mask_neg=(ys<=inlier_th).float(),(ys>inlier_th).float()
    num_pos,num_neg=torch.relu(torch.sum(mask_pos, dim=1) - 1.0) + 1.0,torch.relu(torch.sum(mask_neg, dim=1) - 1.0) + 1.0
    loss_pos,loss_neg=-torch.log(abs(confidence) + 1e-8)*mask_pos,-torch.log(abs(1-confidence)+1e-8)*mask_neg
    classif_loss = torch.mean(loss_pos * 0.5 / num_pos.unsqueeze(-1) + loss_neg * 0.5 / num_neg.unsqueeze(-1),dim=-1)
    classif_loss =classif_loss*batch_mask
    # classif_loss=classif_loss.mean()
    precision = torch.sum((confidence > 0.5).type(confidence.type()) * mask_pos, dim=1) / (torch.sum((confidence > 0.5).type(confidence.type()), dim=1)+1e-8)
    recall = torch.sum((confidence > 0.5).type(confidence.type()) * mask_pos, dim=1) / num_pos
    
    return classif_loss,precision,recall

def weight_loss(log_assignment, weights, gamma=0.0):
    b, m, n = log_assignment.shape
    m -= 1
    n -= 1

    loss_sc = log_assignment * weights

    num_neg0 = weights[:, :m, -1].sum(-1).clamp(min=1.0)
    num_neg1 = weights[:, -1, :n].sum(-1).clamp(min=1.0)
    num_pos = weights[:, :m, :n].sum((-1, -2)).clamp(min=1.0)

    nll_pos = -loss_sc[:, :m, :n].sum((-1, -2))
    nll_pos /= num_pos.clamp(min=1.0)

    nll_neg0 = -loss_sc[:, :m, -1].sum(-1)
    nll_neg1 = -loss_sc[:, -1, :n].sum(-1)

    nll_neg = (nll_neg0 + nll_neg1) / (num_neg0 + num_neg1)

    return nll_pos, nll_neg, num_pos, (num_neg0 + num_neg1) / 2.0


class NLLLoss(nn.Module):
    default_conf = {
        "nll_balancing": 0.5,
        "gamma_f": 0.0,  # focal loss
    }

    def __init__(self, conf):
        super().__init__()
        self.conf = OmegaConf.merge(self.default_conf, conf)
        self.loss_fn = self.nll_loss

    def forward(self, pred, data, weights=None):
        log_assignment = pred["log_assignment"]
        if weights is None:
            weights = self.loss_fn(log_assignment, data)
        nll_pos, nll_neg, num_pos, num_neg = weight_loss(
            log_assignment, weights, gamma=self.conf.gamma_f
        )
        nll = (
            self.conf.nll_balancing * nll_pos + (1 - self.conf.nll_balancing) * nll_neg
        )

        return (
            nll,
            weights,
            {
                "assignment_nll": nll,
                "nll_pos": nll_pos,
                "nll_neg": nll_neg,
                "num_matchable": num_pos,
                "num_unmatchable": num_neg,
            },
        )

    def nll_loss(self, log_assignment, data):
        m, n = data["gt_matches0"].size(-1), data["gt_matches1"].size(-1)
        positive = data["gt_assignment"].float()
        neg0 = (data["gt_matches0"] == -1).float()
        neg1 = (data["gt_matches1"] == -1).float()

        weights = torch.zeros_like(log_assignment)
        weights[:, :m, :n] = positive

        weights[:, :m, -1] = neg0
        weights[:, -1, :m] = neg1
        return weights
