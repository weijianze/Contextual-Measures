#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F



class ClsLoss(nn.Module):
    ''' Classic loss function for face recognition '''

    def __init__(self, args):

        super(ClsLoss, self).__init__()
        self.args     = args


    def forward(self, predy, target, weight = None, mu = None, logvar = None):

        loss = None
        sum_weight = 0.
        if self.args.loss_mode == 'focal_loss':
            logp = F.cross_entropy(predy, target, reduce=False)
            prob = torch.exp(-logp)
            loss = ((1-prob) ** self.args.loss_power * logp).mean()

        elif self.args.loss_mode == 'hardmining':
            batchsize = predy.shape[0]
            logp      = F.cross_entropy(predy, label, reduce=False)
            inv_index = torch.argsort(-logp) # from big to small
            num_hard  = int(self.args.hard_ratio * batch_size)
            hard_idx  = ind_sorted[:num_hard]
            loss      = torch.sum(F.cross_entropy(pred[hard_idx], label[hard_idx]))

        else: # navie-softmax
            # loss = F.cross_entropy(predy, target)
            loss_list = F.cross_entropy(predy, target, reduction="none")
            if (weight is not None):
                loss_list = loss_list * weight
            loss = torch.mean(loss_list)

        if (mu is not None) and (logvar is not None):
            # kl_loss = -(1 + logvar - mu.pow(2) - logvar.exp()) / 2
            ################################################
            kl_loss = -(1 + logvar  - logvar.exp()) / 2
            ################################################
            kl_loss = kl_loss.sum(dim=1).mean()
            loss    = loss + self.args.kl_lambda * kl_loss
            
        return loss


class RegLoss(nn.Module):

    def __init__(self, feat_dim = 512, classnum = 85742):
        super(RegLoss, self).__init__()
        self.feat_dim = feat_dim
        self.classnum = classnum
        self.centers  = torch.Tensor(classnum, feat_dim)
        
        
    def fetch_center_from_fc_layer(self, fc_state_dict):
        weights_key = 'module.weight' if 'module.weight' in fc_state_dict.keys() else 'weight'
        try:
            weights = fc_state_dict[weights_key]
        except Exception as e:
            print(e)
        else:
            assert weights.size() == torch.Size([self.classnum, self.feat_dim]), \
                'weights.size can not match with (classnum, feat_dim)'
            self.centers = weights
            print('Fetch the center from fc-layer was finished ...')

            
    def forward(self, mu, logvar, labels):
        fit_loss = (self.fc_weights[labels] - mu).pow(2) / (1e-10 + torch.exp(logvar))
        reg_loss = (fit_loss + logvar) / 2.0
        reg_loss = torch.sum(reg_loss, dim=1).mean()
        return reg_loss
