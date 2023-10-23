"""
Author: Yonglong Tian (yonglong@mit.edu)
Date: May 07, 2020
"""
from __future__ import print_function

import torch
import torch.nn as nn


class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):    
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0] 
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1) # 8,1
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)   # unbind creates tuple of tensors. Tuple length=2 i.e. tuple has 'n_views' tensors of shape bsz,channel_dim
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        anchor_dot_contrast = torch.div(
            torch.matmul(contrast_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos 
        # just return loss.mean()
        loss = loss.view(anchor_count, batch_size).mean()

        return loss


'''
bsz = 2
feature_dim = 512
feature_dim = 256
n_views = 2
features = torch.rand(bsz,n_views,feature_dim)
eatures.shape
torch.Size([2, 2, 256])
features
tensor([[[0.7705, 0.1416, 0.2262,  ..., 0.3963, 0.9758, 0.0711],
         [0.4353, 0.9371, 0.1170,  ..., 0.3693, 0.4035, 0.2493]],

        [[0.0489, 0.7539, 0.2394,  ..., 0.5251, 0.7075, 0.1050],
         [0.5052, 0.2401, 0.2846,  ..., 0.0054, 0.2589, 0.5959]]])
labels = torch.rand(bsz)
labels.shape
torch.Size([2])
labels
tensor([0.7336, 0.8206])
labels[0] = 2
labels[1] = 1
labels
tensor([2., 1.])
mask = torch.eq(labels, labels.T).float().to(device)
mask.shape
torch.Size([2])
mask
tensor([1., 1.], device='cuda:0')
labels
tensor([2., 1.])
labels.T
tensor([2., 1.])
labels = labels.contiguous().view(-1, 1)
labels.shape
torch.Size([2, 1])
mask = torch.eq(labels, labels.T).float()
mask
tensor([[1., 0.],
        [0., 1.]])
labels.contiguous().view(-1, 1)
tensor([[2.],
        [1.]])

contrast_count = features.shape[1]
contrast_count
2
tmp = torch.unbind(features,dim=1)
features.shape
torch.Size([2, 2, 256])
mp
(tensor([[7.7051e-01,...503e-01]]), tensor([[4.3527e-01,...586e-01]]))
tmp[0].shape
torch.Size([2, 256])
torch.cat(tmp,dim=0)
tensor([[0.7705, 0.1416, 0.2262,  ..., 0.3963, 0.9758, 0.0711],
        [0.0489, 0.7539, 0.2394,  ..., 0.5251, 0.7075, 0.1050],
        [0.4353, 0.9371, 0.1170,  ..., 0.3693, 0.4035, 0.2493],
        [0.5052, 0.2401, 0.2846,  ..., 0.0054, 0.2589, 0.5959]])

contrast_feature = torch.cat(tmp,dim=0)
contrast_feature.shape
torch.Size([4, 256])
contrast_count = features.shape[1]
contrast_count
2
temperature = 1
anchor_dot_contrast = torch.div(
            torch.matmul(contrast_feature, contrast_feature.T),
            temperature)
anchor_dot_contrast.shape
torch.Size([4, 4])
 torch.max(anchor_dot_contrast, dim=1, keepdim=True)
torch.return_types.max(
values=tensor([[79.9767],
        [88.6778],
        [89.3391],
        [88.2184]]),
indices=tensor([[0],
        [1],
        [2],
        [3]]))

logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
logits_max.shape
torch.Size([4, 1])
anchor_dot_contrast - logits_max.detach()
tensor([[  0.0000, -15.8391, -17.4363, -21.0205],
        [-24.5402,   0.0000, -19.0478, -21.5090],
        [-26.7987, -19.7091,   0.0000, -23.4159],
        [-29.2622, -21.0495, -22.2952,   0.0000]])
logits = anchor_dot_contrast - logits_max.detach()
logits.shape
torch.Size([4, 4])
logits
tensor([[  0.0000, -15.8391, -17.4363, -21.0205],
        [-24.5402,   0.0000, -19.0478, -21.5090],
        [-26.7987, -19.7091,   0.0000, -23.4159],
        [-29.2622, -21.0495, -22.2952,   0.0000]])


mask = mask.repeat(contrast_count, contrast_count)
mask.shape
torch.Size([4, 4])
mask
tensor([[1., 0., 1., 0.],
        [0., 1., 0., 1.],
        [1., 0., 1., 0.],
        [0., 1., 0., 1.]])

torch.arange(bsz * contrast_count)
tensor([0, 1, 2, 3])
tmp2= torch.arange(bsz * contrast_count)
tmp2
tensor([0, 1, 2, 3])
torch.ones_like(mask)
tensor([[1., 1., 1., 1.],
        [1., 1., 1., 1.],
        [1., 1., 1., 1.],
        [1., 1., 1., 1.]])
tmp2 = tmp2.view(-1,1)

logits_mask =  torch.scatter(torch.ones_like(mask),1,tmp2,0)
logits_mask
tensor([[0., 1., 1., 1.],
        [1., 0., 1., 1.],
        [1., 1., 0., 1.],
        [1., 1., 1., 0.]])
mask
tensor([[1., 0., 1., 0.],
        [0., 1., 0., 1.],
        [1., 0., 1., 0.],
        [0., 1., 0., 1.]])
mask = mask * logits_mask
mask
tensor([[0., 0., 1., 0.],
        [0., 0., 0., 1.],
        [1., 0., 0., 0.],
        [0., 1., 0., 0.]])
orch.exp(logits) * logits_mask
tensor([[0.0000e+00, 1.3218e-07, 2.6763e-08, 7.4290e-10],
        [2.1995e-11, 0.0000e+00, 5.3410e-09, 4.5580e-10],
        [2.2987e-12, 2.7569e-09, 0.0000e+00, 6.7703e-11],
        [1.9571e-13, 7.2161e-10, 2.0765e-10, 0.0000e+00]])
exp_logits = torch.exp(logits) * logits_mask
log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
log_prob
tensor([[15.6500, -0.1890, -1.7862, -5.3704],
        [-5.5781, 18.9622, -0.0857, -2.5468],
        [-7.1146, -0.0251, 19.6841, -3.7318],
        [-8.4657, -0.2531, -1.4987, 20.7964]])
mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
mean_log_prob_pos
tensor([-1.7862, -2.5468, -7.1146, -0.2531])
loss = - mean_log_prob_pos
loss
tensor([1.7862, 2.5468, 7.1146, 0.2531])
loss.view(contrast_count,bsz).mean()
tensor(2.9252)

'''
