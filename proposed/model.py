from collections import namedtuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from dtw import SoftDTW


def xavier_uniform(layer, w_init_gain='linear', outs=None):
    if outs is None:
        nn.init.xavier_uniform_(layer.weight, gain=nn.init.calculate_gain(w_init_gain))
    else:
        c = 0
        for out in outs:
            nn.init.xavier_uniform_(layer.weight[c:c+out, ...], gain=nn.init.calculate_gain(w_init_gain))
            c += out
    return layer


def wn_xavier(layer, w_init_gain='linear', outs=None):
    return nn.utils.weight_norm(xavier_uniform(layer, w_init_gain=w_init_gain, outs=outs))


def kl_normal(x, eps=1e-4):
    assert len(x.size()) == 3
    mean = x.mean(dim=2, keepdim=True)
    diff = x-mean
    sigma = diff.matmul(diff.transpose(1,2))/x.size(2) + eps*torch.eye(x.size(1), dtype=x.dtype, device=x.device)[None, :, :]
    kl = -sigma.logdet().sum() + sigma.diagonal(dim1=1, dim2=2).sum() + mean.pow(2).sum()
    return 0.5*(kl - x.size(1))/x.size(0)/x.size(1)

def modulation_spec(x):
    s = torch.rfft(x, 1, normalized=True)
    a = s[..., 0]**2 + s[..., 1]**2
    return a

def remove(conv_list):
    new_conv_list = torch.nn.ModuleList()
    for old_conv in conv_list:
        old_conv = torch.nn.utils.remove_weight_norm(old_conv)
        new_conv_list.append(old_conv)
    return new_conv_list


class Attention(nn.Module):
    def __init__(self, n_head=1, alpha=5.0):
        super().__init__()
        self.n_head = n_head
        self.alpha = alpha

    def forward(self, q, k, v, mask=None):
        # q: bxdxm
        # k: bxdxn
        # v: bxcxn
        # mask: bxmxn | 1xmxn
        # ret: bxcxm
        n_head = self.n_head
        q_shape = q.size()
        v_shape = v.size()
        if n_head > 1:
            v = v.view(*v_shape[:-2], n_head, v_shape[-2]//n_head, v_shape[-1])
        sm = self.softmax(q, k, alpha=self.alpha, mask=mask)
        ret = (sm.unsqueeze(-3) * v.unsqueeze(-2)).sum(dim=-1) # bxcxmxn -> bxcxm
        if n_head > 1:
            ret = ret.view(-1, v_shape[-2], q_shape[-1])
        return ret

    def softmax(self, q, k, alpha=1.0, mask=None):
        n_head = self.n_head
        q_shape = q.size()
        k_shape = k.size()
        if n_head > 1:
            q, k = q.contiguous(), k.contiguous()
            q = q.view(*q_shape[:-2], n_head, q_shape[-2]//n_head, q_shape[-1])
            k = k.view(*k_shape[:-2], n_head, k_shape[-2]//n_head, k_shape[-1])
            if mask is not None:
                mask = mask.unsqueeze(-3)
        q = F.normalize(q, dim=-2)
        k = F.normalize(k, dim=-2)
        qk = self.alpha * torch.matmul(q.transpose(-1, -2), k) # bxhxmxn
        if mask is not None:
            return F.softmax(mask+qk, dim=-1) # bxmxn
        else:
            return F.softmax(qk, dim=-1) # bxmxn


class UEncoder(nn.Module):
    def __init__(self, c_in=40, c_h=64, c_out=2, c_k=64, n_layers=5):
        super().__init__()
        self.c_in = c_in
        self.c_h = c_h
        self.c_out = c_out
        self.c_k = c_k
        self.n_layers = n_layers
        
        self.contract = nn.ModuleList()
        self.expand = nn.ModuleList()
        L = n_layers
        # contract
        self.contract.append(wn_xavier(nn.Conv1d(c_in, c_h, 3)))
        self.contract.append(wn_xavier(nn.Conv1d(c_h, c_h, 1)))
        for n in range(L-1):
            self.contract.append(wn_xavier(nn.Conv1d(c_h, c_h, 3)))
            self.contract.append(wn_xavier(nn.Conv1d(c_h, c_h, 1)))
        # expand
        for n in range(L-1):
            self.expand.append(wn_xavier(nn.Conv1d(c_h, c_h, 3)))
            self.expand.append(wn_xavier(nn.Conv1d(c_h, c_h+c_k+c_out*2**(L-n-1), 1), outs=[c_h, c_k, c_out*2**(L-n-1)]))
        self.expand.append(wn_xavier(nn.Conv1d(c_h, c_h, 3)))
        self.expand.append(wn_xavier(nn.Conv1d(c_h, c_out + c_k, 1), outs=[c_k, c_out]))
        
    def forward(self, x):
        L = self.n_layers
        skips = []
        outs = []
        rs = x

        for n in range(L):
            if n >= 1:
                rs = F.avg_pool1d(rs, 2, 2)
            skip = F.pad(rs, [1, 1], 'reflect')
            skip = self.contract[2*n](skip)
            skip = F.gelu(skip)
            skip = self.contract[2*n+1](skip)
            skips.append(skip)
            if n==0:
                rs = F.gelu(skip)
            else:
                rs = rs + F.gelu(skip)
        skips = list(reversed(skips))
        for n in range(L):
            skip = F.pad(rs, [1, 1], 'reflect')
            skip = self.expand[2*n](skip)
            skip = F.gelu(skip)
            skip = skip + skips[n]
            skip = self.expand[2*n+1](skip)
            if n < L-1:
                skip, out = skip.split([self.c_h, self.c_k+self.c_out*2**(L-n-1)], dim=1)
            else:
                out = skip
            outs.append(out)
            
            if n < L-1:
                rs = rs + F.gelu(skip)
                rs = F.interpolate(rs, scale_factor=2)
        return outs

    def remove_wn(self):
        self.contract = remove(self.contract)
        self.expand = remove(self.expand)
                

class UDecoder(nn.Module):
    def __init__(self, c_w=2, c_z=2, c_k=64, c_h=64, c_out=40, n_head=4, attention_alpha=5.0, n_layers=5):
        super().__init__()
        self.c_w = c_w
        self.c_z = c_z
        self.c_k = c_k
        self.c_h = c_h
        self.c_out = c_out
        self.n_head = n_head
        self.attention_alpha = attention_alpha
        self.n_layers = n_layers
        self.convs = nn.ModuleList()
        self.inputs = nn.ModuleList()
        self.conds = nn.ModuleList()
        self.output = wn_xavier(nn.Conv1d(c_h, c_out, 1))
        self.attention = Attention(alpha=attention_alpha, n_head=n_head)
        L = n_layers
        for n in range(L):
            self.convs.append(wn_xavier(nn.Conv1d(c_h, c_h, 3)))
            self.convs.append(wn_xavier(nn.Conv1d(c_h, c_h, 1)))
            self.inputs.append(wn_xavier(nn.Conv1d(c_w*2**(L-n-1), c_h, 1)))
            self.conds.append(wn_xavier(nn.Conv1d(c_z*2**(L-n-1), c_h, 1)))
            
    def forward(self, qws, kvs, use_w=False):
        L = self.n_layers
        ws = []
        for n in range(L):
            q, w = qws[n].split([self.c_k, self.c_w*2**(L-n-1)], dim=1)
            ws.append(w)
            k, v = kvs[n].split([self.c_k, self.c_z*2**(L-n-1)], dim=1)
            z = self.attention(q, k, v)
            
            if n==0:
                rs = self.inputs[n](w)
            else:
                rs = rs + self.inputs[n](w)
            
            skip = F.pad(rs, [1, 1], 'reflect')
            skip = self.convs[2*n](skip)
            skip = F.gelu(skip)
            skip = skip + self.conds[n](z)
            skip = self.convs[2*n+1](skip)
            rs = rs + F.gelu(skip)
            if n < L-1:
                rs = F.interpolate(rs, scale_factor=2)
                
        if use_w:
            return self.output(rs), ws
        else:
            return self.output(rs)

    def remove_wn(self):
        self.output = torch.nn.utils.remove_weight_norm(self.output)
        self.convs = remove(self.convs)
        self.inputs = remove(self.inputs)
        self.conds = remove(self.conds)

    def softmax(self, qws, kvs):
        L = self.n_layers
        sms = []
        for n in range(L):
            q, w = qws[n].split([self.c_k, self.c_w*2**(L-n-1)], dim=1)
            k, v = kvs[n].split([self.c_k, self.c_z*2**(L-n-1)], dim=1)
            sm = self.attention.softmax(q, k)
            sms.append(sm)
        return sms


class VC(nn.Module):
    def __init__(self, config, train=True):
        super().__init__()
        self.speaker_encoder = UEncoder(**config['SpeakerEncoder'])
        self.content_encoder = UEncoder(**config['ContentEncoder'])
        self.decoder = UDecoder(**config['Decoder'])
        
        if train:
            self.lambda_kl = config['lambda']['lambda_kl']
            self.lambda_mse = config['lambda']['lambda_mse']
            self.lambda_dtw = config['lambda']['lambda_dtw']
            self.dtw = SoftDTW(**config['dtw'])
            
    def forward(self, x1, x2, y1, y2, z1, z2):
        loss_dtw, loss_kl_p = self.parallel(x1, x2, y1, y2)
        loss_mse, loss_kl_np, loss_ms_np = self.nonpara(z1, z2)
        return self.lambda_dtw*loss_dtw + self.lambda_mse*loss_mse + self.lambda_kl*(loss_kl_p+loss_kl_np) + 1e-1*loss_ms_np, (loss_dtw, loss_mse, loss_kl_p, loss_kl_np, loss_ms_np)
    
    def nonpara(self, z1, z2):
        # print(z1.size())
        # print(z2.size())
        qws1 = self.content_encoder(z1)
        kvs1 = self.speaker_encoder(z1)
        qws2 = self.content_encoder(z2)
        kvs2 = self.speaker_encoder(z2)
        r1, ws1 = self.decoder(qws1, kvs2, use_w=True)
        r2, ws2 = self.decoder(qws2, kvs1, use_w=True)
        loss_mse = 0.5*((z1-r1).pow(2).mean() + (z2-r2).pow(2).mean())
        loss_ms = torch.sqrt((modulation_spec(z1)-modulation_spec(r1)).pow(2).mean() + (modulation_spec(z2)-modulation_spec(r2)).pow(2).mean())
        loss_kl = 0.5*sum([kl_normal(w) for w in ws1+ws2])/len(ws1)
        return loss_mse, loss_kl, loss_ms

    def parallel(self, x1, x2, y1, y2):
        batch = [self.pad(x) for x in [x1, x2, y1, y2]]
        codes = [(self.content_encoder(z), self.speaker_encoder(z)) for z in batch]
        rws = [self.decoder(x[0], r[1], use_w=True) for x, r in zip(codes, reversed(codes))] # (x1, y2), (x2, y1), (y1, x2), (y2, x1)
        gts = [*batch[-2:], *batch[:2]] # y1, y2, x1, x2
        loss_dtw = 0.25 * sum([self.dtw(z1, z2).sum()/z1.numel() for (z1, _), z2 in zip(rws, gts)])
        loss_kl = 0.25 * sum([sum([kl_normal(w2) for w2 in w])/len(w) for (_, w) in rws])
        return loss_dtw, loss_kl

    def pad(self, x):
        t = 16
        t_size = x.size(-1)
        if t_size % t == 0:
            return x
        else:
            return F.pad(x, (0, t-t_size%t))

    def remove_wn(self):
        self.speaker_encoder.remove_wn()
        self.content_encoder.remove_wn()
        self.decoder.remove_wn()

    @torch.no_grad()
    def make_input(self, mel):
        mel = torch.as_tensor(mel)
        if len(mel.size()) == 2:
            mel = mel.unsqueeze(0)
        assert len(mel.size()) == 3
        if next(self.parameters()).is_cuda:
            mel = mel.to(torch.device('cuda'))
        return self.pad(mel)

    @torch.no_grad()
    def content(self, mel):
        mel = self.make_input(mel)
        cq = self.content_encoder(mel)
        return cq

    @torch.no_grad()
    def extract(self, mel):
        mel = self.make_input(mel)
        kv = self.speaker_encoder(mel)
        return kv
    
    @torch.no_grad()
    def reconstruct_mcep(self, cq, kv):
        rec = self.decoder(cq, kv)
        return rec
    
    @torch.no_grad()
    def convert_all(self, cq, kv):
        return self.decoder.convert(cq, kv)

    @torch.no_grad()
    def convert(self, mel, ref):
        cq = self.content(mel)
        kv = self.extract(ref)
        return self.reconstruct_mcep(cq, kv)
