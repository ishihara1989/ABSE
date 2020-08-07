import os.path
from pathlib import Path

import numpy as np
import torch
from numba import jit
from torch.autograd import Function

from torch.utils.cpp_extension import load

class Loader():
    def __init__(self):
        self.DTW_CUDA_SUCCEED = False
        self.dtw_cuda = None
    def load(self):
        if self.dtw_cuda is None:
            try:
                root = Path(__file__).resolve().parent
                self.dtw_cuda = load(name='dtw_cuda', sources=[root/'soft_dtw_src/dtw_cuda.cpp', root/'soft_dtw_src/dtw_cuda_kernel.cu'])
                self.DTW_CUDA_SUCCEED = True
            except Exception as e:
                self.DTW_CUDA_SUCCEED = False
                print(e)
                print('cuda DTW load failed')

loader = Loader()

@jit(nopython = True)
def compute_softdtw(D, gamma):
    B = D.shape[0]
    N = D.shape[1]
    M = D.shape[2]
    R = np.ones((B, N + 2, M + 2)) * np.inf
    R[:, 0, 0] = 0
    for k in range(B):
        for j in range(1, M + 1):
            for i in range(1, N + 1):
                r0 = -R[k, i - 1, j - 1] / gamma
                r1 = -R[k, i - 1, j] / gamma
                r2 = -R[k, i, j - 1] / gamma
                rmax = max(max(r0, r1), r2)
                rsum = np.exp(r0 - rmax) + np.exp(r1 - rmax) + np.exp(r2 - rmax)
                softmin = - gamma * (np.log(rsum) + rmax)
                R[k, i, j] = D[k, i - 1, j - 1] + softmin
    return R

@jit(nopython = True)
def compute_softdtw_backward(D_, R, gamma):
    B = D_.shape[0]
    N = D_.shape[1]
    M = D_.shape[2]
    D = np.zeros((B, N + 2, M + 2))
    E = np.zeros((B, N + 2, M + 2))
    D[:, 1:N + 1, 1:M + 1] = D_
    E[:, -1, -1] = 1
    R[:, : , -1] = -np.inf
    R[:, -1, :] = -np.inf
    R[:, -1, -1] = R[:, -2, -2]
    for k in range(B):
        for j in range(M, 0, -1):
            for i in range(N, 0, -1):
                a0 = (R[k, i + 1, j] - R[k, i, j] - D[k, i + 1, j]) / gamma
                b0 = (R[k, i, j + 1] - R[k, i, j] - D[k, i, j + 1]) / gamma
                c0 = (R[k, i + 1, j + 1] - R[k, i, j] - D[k, i + 1, j + 1]) / gamma
                a = np.exp(a0)
                b = np.exp(b0)
                c = np.exp(c0)
                E[k, i, j] = E[k, i + 1, j] * a + E[k, i, j + 1] * b + E[k, i + 1, j + 1] * c
    return E[:, 1:N + 1, 1:M + 1]

class _SoftDTW(Function):
    '''
    from https://github.com/Sleepwalking/pytorch-softdtw
    '''
    @staticmethod
    def forward(ctx, D, gamma):
        dev = D.device
        dtype = D.dtype
        gamma = torch.Tensor([gamma]).to(dev).type(dtype) # dtype fixed
        D_ = D.detach().cpu().numpy()
        g_ = gamma.item()
        R = torch.Tensor(compute_softdtw(D_, g_)).to(dev).type(dtype)
        ctx.save_for_backward(D, R, gamma)
        return R[:, -2, -2]

    @staticmethod
    def backward(ctx, grad_output):
        dev = grad_output.device
        dtype = grad_output.dtype
        D, R, gamma = ctx.saved_tensors
        D_ = D.detach().cpu().numpy()
        R_ = R.detach().cpu().numpy()
        g_ = gamma.item()
        E = torch.Tensor(compute_softdtw_backward(D_, R_, g_)).to(dev).type(dtype)
        return grad_output.view(-1, 1, 1).expand_as(E) * E, None


class _SoftDTWCuda(Function):
    @staticmethod
    def forward(ctx, D, gamma):
        R = loader.dtw_cuda.forward(D, gamma)
        ctx.save_for_backward(D, R, torch.as_tensor(gamma))
        return R[:, -1, -1]

    @staticmethod
    def backward(ctx, grad_output):
        E = loader.dtw_cuda.backward(*ctx.saved_tensors)
        return grad_output[:, None, None] * E, None

class SoftDTW(torch.nn.Module):
    def __init__(self, gamma=1.0, normalize=False, cuda=True, save_memory=True):
        super(SoftDTW, self).__init__()
        if cuda:
            loader.load()
        self.use_cuda = cuda
        self.normalize = normalize
        self.gamma = gamma
        self.save_memory = save_memory

    def calc_distance_matrix(self, x, y):
        if self.save_memory:
            xx = x.pow(2).sum(dim=1) # S, T
            yy = y.pow(2).sum(dim=1) # S, T'
            xy = x.transpose(1,2).matmul(y) # S, T, T'
            pdist = xx[:,:,None]+yy[:,None,:]-2*xy
            return pdist
        else:
            return torch.pow(x[:, :, :, None]- y[:, :, None, :], 2).sum(dim=1)

    def forward(self, x, y):
        func_dtw = _SoftDTWCuda.apply if loader.DTW_CUDA_SUCCEED and self.use_cuda and x.is_cuda else _SoftDTW.apply
        assert len(x.shape) == len(y.shape)
        squeeze = False
        if len(x.shape) < 3:
            x = x.unsqueeze(0)
            y = y.unsqueeze(0)
        squeeze = True
        if self.normalize:
            D_xy = self.calc_distance_matrix(x, y)
            out_xy = func_dtw(D_xy, self.gamma)
            D_xx = self.calc_distance_matrix(x, x)
            out_xx = func_dtw(D_xx, self.gamma)
            D_yy = self.calc_distance_matrix(y, y)
            out_yy = func_dtw(D_yy, self.gamma)
            result = out_xy - 1/2 * (out_xx + out_yy) # distance
        else:
            D_xy = self.calc_distance_matrix(x, y)
            out_xy = func_dtw(D_xy, self.gamma)
            result = out_xy # discrepancy
        return result.squeeze(0) if squeeze else result


if __name__ == "__main__":
    x = torch.randn(1,4,513, requires_grad=True)
    y = torch.randn(1,4,1027, requires_grad=True)
    loss_fun = SoftDTW()
    loss = loss_fun(x, y)
    print(loss)
    loss.backward()
    print(x.grad)
    print(y.grad)

    x.grad=None
    y.grad=None
    loss = loss_fun(x.cuda(), y.cuda())
    loss.backward()
    print(loss)
    print(x.grad)
    print(y.grad)

    xy=loss_fun.calc_distance_matrix(x, y)
    loss_fun.save_memory=False
    xy2=loss_fun.calc_distance_matrix(x, y)
    print((xy-xy2).pow(2).sum())
