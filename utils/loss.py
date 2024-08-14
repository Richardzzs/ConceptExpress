import torch
import torch.nn as nn
import torch.nn.functional as F
from ptp_utils import kld_distance

class SupConLoss(nn.Module):
    def __init__(self, temperature=0.07, contrast_mode='one',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))
        # # torch.Size([10, 1, 1024])
        # print("="*30)
        # print("***features 1***")
        # print(type(features))
        # print(features.shape)
        # print(features)
        # print("="*30)

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)
        # # torch.Size([10, 1, 1024]), 这里的feature与上面的feature 1没有变化，因为feature的维度就是3
        # print("="*30)
        # print("***features 2***")
        # print(type(features))
        # print(features.shape)
        # print(features)
        # print("="*30)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1) #  一维张量转化为二维张量
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)  # 创建一个与 labels 相同大小的布尔矩阵 mask，True False转化为1 0
        else:
            mask = mask.float().to(device)
        # # torch.Size([10, 10])
        # print("="*30)
        # print("mask")
        # print(type(mask))
        # print(mask.shape)
        # print(mask)
        # print("="*30)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        # # torch.Size([10, 1024])
        # print("="*30)
        # print("***contrast_feature***")
        # print(type(contrast_feature))
        # print(contrast_feature.shape)
        # print(contrast_feature)
        # print("="*30)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))
        # print("="*30)
        # # 1 1
        # print("***COUNT***")
        # print(type(anchor_count))
        # print(type(contrast_count))
        # print(anchor_count)
        # print(contrast_count)
        # print("="*30)
        # # torch.Size([10, 1024])
        # print("="*30)
        # print("***anchor_feature***")
        # print(type(anchor_feature))
        # print(anchor_feature.shape)
        # print(anchor_feature)
        # print("="*30)

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # # torch.Size([10, 10])
        # print("="*30)
        # print("***anchor_dot_contrast***")
        # print(type(anchor_dot_contrast))
        # print(anchor_dot_contrast.shape)
        # print(anchor_dot_contrast)
        # print("="*30)
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
        exp_logits_d = torch.exp(logits) * logits_mask
        exp_logits_n = torch.exp(logits) * mask
        log_prob = torch.log(exp_logits_n.sum(1, keepdim=True)) - torch.log(exp_logits_d.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = log_prob.sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()  # normalize
        # print(loss)
        return loss
    

class SupKLDiergence(nn.Module):
    def __init__(self, contrast_mode='one'):
        self.contrast_mode = contrast_mode
        super(SupKLDiergence, self).__init__()

    def forward(self, features, labels=None, mask=None):
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
            labels = labels.contiguous().view(-1, 1) #  一维张量转化为二维张量
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)  # 创建一个与 labels 相同大小的布尔矩阵 mask，True False转化为1 0
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # 将 v 转换为概率分布，使用 softmax 函数
        prob_dist = F.softmax(anchor_feature, dim=1).to(device)
        
        # 将 label 转换为一个索引张量，用于后续操作
        label_indices = torch.unique(labels, return_inverse=True)[1].to(device)

        # 初始化相同标签的KL散度之和
        same_label_kl_sum = torch.tensor(0).to(device)

        # 对于每个标签组，计算KL散度之和
        for i in torch.unique(labels):
            # 获取当前标签的索引
            indices = (label_indices == i).nonzero(as_tuple=True)[0]
            
            # 计算当前标签组内所有点对的KL散度之和
            for j in range(len(indices)):
                for k in range(j + 1, len(indices)):
                    p = prob_dist[indices[j]]
                    q = prob_dist[indices[k]]
                    kl_div = torch.div((F.kl_div(torch.log(p), q, reduction='batchmean') + F.kl_div(torch.log(q), p, reduction='batchmean')), 2)
                    same_label_kl_sum = torch.add(same_label_kl_sum, kl_div)

        # # 转换为常量
        # same_label_kl_sum = same_label_kl_sum.item()

        # 初始化不同标签的KL散度之和
        diff_label_kl_sum = torch.tensor(0).to(device)

        # 遍历所有点对，计算不同标签的KL散度之和
        for i in range(len(anchor_feature)):
            for j in range(i + 1, len(anchor_feature)):
                if labels[i] != labels[j]:  # 确保点对的标签不同
                    p = prob_dist[i]
                    q = prob_dist[j]
                    kl_div = torch.div((F.kl_div(torch.log(p), q, reduction='batchmean') + F.kl_div(torch.log(q), p, reduction='batchmean')), 2)
                    diff_label_kl_sum = torch.add(diff_label_kl_sum, kl_div)
        # 转换为常量
        # diff_label_kl_sum = diff_label_kl_sum.item()

        loss = torch.div(same_label_kl_sum, diff_label_kl_sum)
        # print("@"*30)
        print(loss)
        # print("@"*30)
        return loss