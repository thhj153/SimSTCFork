import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
import numpy as np

class Classifier(nn.Module):
    def __init__(self, in_fea, hid_fea, out_fea, drop_out=0.5):
        super(Classifier, self).__init__()
        self.projector = nn.Sequential(
            nn.Linear(in_fea, hid_fea),
            nn.BatchNorm1d(hid_fea),
            nn.ReLU(inplace=True),
            nn.Linear(hid_fea, out_fea))

    def forward(self, doc_fea):
        z = F.normalize(self.projector(doc_fea),dim=1)
        return z


class UCL(nn.Module):
    def __init__(self, in_fea, out_fea, temperature=0.5):
        super(UCL, self).__init__()
        self.projector = nn.Sequential(
            nn.Linear(in_fea, out_fea),
            nn.BatchNorm1d(out_fea),
            nn.ReLU(inplace=True),
            nn.Linear(out_fea, out_fea))
        self.projector_2 = nn.Sequential(
            nn.Linear(in_fea+300, out_fea),
            nn.BatchNorm1d(out_fea),
            nn.ReLU(inplace=True),
            nn.Linear(out_fea, out_fea))

        self.tem = temperature
        self.hidden_fea = in_fea
    
    def sim(self, z1: torch.Tensor, z2: torch.Tensor):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())

    def forward(self, doc_fea):
        total_loss = 0
        for i in range(2):
            for j in range(i+1, 3):
                out_1 = self.projector(doc_fea[i]) if doc_fea[i].shape[1] == self.hidden_fea else self.projector_2(doc_fea[i])
                out_2 = self.projector(doc_fea[j]) if doc_fea[j].shape[1] == self.hidden_fea else self.projector_2(doc_fea[j])
                out_1, out_2 = F.normalize(out_1, dim=1), F.normalize(out_2, dim=1)
                
                out = torch.cat([out_1, out_2], dim=0)
                dim = out.shape[0]

                batch_size = 5120 #* 2 #2560
                l1 = self.batch_loss(out_1, out_2, batch_size)
                l2 = self.batch_loss(out_2, out_1, batch_size)

                loss = (l1+l2) * 0.5
                total_loss += loss.mean()
                return total_loss
    
    def batch_loss(self, out_1, out_2, batch_size):
        device = out_1.device
        num_nodes = out_1.size(0)
        num_batches = (num_nodes - 1) // batch_size + 1
        f = lambda x: torch.exp(x / self.tem)
        indices = torch.arange(0, num_nodes).to(device)
        losses = []

        for i in range(num_batches):
            mask = indices[i * batch_size:(i + 1) * batch_size]
            refl_sim = f(self.sim(out_1[mask], out_1))  # [B, N]
            between_sim = f(self.sim(out_1[mask], out_2))  # [B, N]

            losses.append(-torch.log(
                between_sim[:, i * batch_size:(i + 1) * batch_size].diag()
                / (refl_sim.sum(1) + between_sim.sum(1)
                   - refl_sim[:, i * batch_size:(i + 1) * batch_size].diag())))

        return torch.cat(losses)
