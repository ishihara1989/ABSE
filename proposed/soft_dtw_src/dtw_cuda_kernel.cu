#include <cmath>
#include <limits>
#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>


template <typename scalar_t>
__global__ void soft_dtw_cuda_forward_kernel(
    const torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> distance,
    torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> R,
    scalar_t gamma
){
    const auto B = blockIdx.x;
    const auto I = threadIdx.x;
    const auto x_size = distance.size(1);
    const auto y_size = distance.size(2);
    const auto T = x_size + y_size;
    for(int ij=0; ij<T; ij++){
        for(int i=I; i<T; i+=blockDim.x){
            const int j = ij-i;
            if(i==0){
                if(j==0){
                    R[B][i][j] = distance[B][i][j];
                }
                else if(j>0 && j<y_size){
                    R[B][i][j] = R[B][i][j-1] + distance[B][i][j];
                }
            }
            else if(i < x_size){
                if(j==0){
                    R[B][i][j] = R[B][i-1][j] + distance[B][i][j];
                }
                else if(j>0 && j<y_size){
                    scalar_t rx = -R[B][i-1][j]/gamma;
                    scalar_t ry = -R[B][i][j-1]/gamma;
                    scalar_t rxy = -R[B][i-1][j-1]/gamma;
                    scalar_t rmax = max(rx, max(ry, rxy));
                    scalar_t rsum = exp(rx-rmax)+exp(ry-rmax)+exp(rxy-rmax);
                    scalar_t softmin = -gamma * (log(rsum) + rmax);
                    R[B][i][j] = distance[B][i][j] + softmin;
                }
            }
        }
        __syncthreads();
    }
}

template <typename scalar_t>
__global__ void soft_dtw_cuda_backward_kernel(
    const torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> distance,
    const torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> R,
    torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> E,
    scalar_t gamma
){
    auto B = blockIdx.x;
    auto I = threadIdx.x;
    auto x_size = distance.size(1);
    auto y_size = distance.size(2);
    auto T = x_size + y_size;
    for(int ij=0; ij<T; ij++){
        for(int i=I; i<T; i+=blockDim.x){
            int j = ij-i;
            if(i==0){
                if(j==0){
                    E[B][x_size-1-i][y_size-1-j] = 1.0;
                }
                else if(j>0 && j<y_size){
                    scalar_t y0 = (R[B][x_size-1-i][y_size-j] - R[B][x_size-1-i][y_size-1-j] - distance[B][x_size-1-i][y_size-j])/gamma;
                    scalar_t gy = exp(y0);
                    E[B][x_size-1-i][y_size-1-j] = gy * E[B][x_size-1-i][y_size-j];
                }
            }
            else if(i < x_size){
                if(j==0){
                    scalar_t x0 = (R[B][x_size-i][y_size-1-j] - R[B][x_size-1-i][y_size-1-j] - distance[B][x_size-i][y_size-1-j])/gamma;
                    scalar_t gx = exp(x0);
                    E[B][x_size-1-i][y_size-1-j] = gx * E[B][x_size-i][y_size-1-j];
                }
                else if(j>0 && j<y_size){
                    scalar_t x0 = (R[B][x_size-i][y_size-1-j] - R[B][x_size-1-i][y_size-1-j] - distance[B][x_size-i][y_size-1-j])/gamma;
                    scalar_t y0 = (R[B][x_size-1-i][y_size-j] - R[B][x_size-1-i][y_size-1-j] - distance[B][x_size-1-i][y_size-j])/gamma;
                    scalar_t xy0 = (R[B][x_size-i][y_size-j] - R[B][x_size-1-i][y_size-1-j] - distance[B][x_size-i][y_size-j])/gamma;
                    scalar_t gx = exp(x0);
                    scalar_t gy = exp(y0);
                    scalar_t gxy = exp(xy0);
                    E[B][x_size-1-i][y_size-1-j] = gx * E[B][x_size-i][y_size-1-j] + gy * E[B][x_size-1-i][y_size-j] + gxy * E[B][x_size-i][y_size-j];
                }
            }
        }
        __syncthreads();
    }
}


torch::Tensor soft_dtw_cuda_forward(torch::Tensor distance, float gamma){
    int B = distance.size(0);
    int N = distance.size(1);
    int M = distance.size(2);
    auto R = torch::zeros({B, N, M},
        torch::dtype(distance.dtype())
        .layout(torch::kStrided)
        .device(distance.device())
        .requires_grad(false));

    auto blocks = B;
    auto threads = 512;
    AT_DISPATCH_FLOATING_TYPES(distance.type(), "soft_dtw_forward_cuda", ([&] {
        soft_dtw_cuda_forward_kernel<scalar_t><<<blocks,threads>>>(
            distance.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
            R.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
            gamma
        );
    }));

    return R;
}

torch::Tensor soft_dtw_cuda_backward(torch::Tensor distance, torch::Tensor R, float gamma){
    int B = distance.size(0);
    int N = distance.size(1);
    int M = distance.size(2);
    auto E = torch::zeros({B, N, M},
        torch::dtype(distance.dtype())
        .layout(torch::kStrided)
        .device(distance.device())
        .requires_grad(false));

    auto blocks = B;
    auto threads = 512;
    AT_DISPATCH_FLOATING_TYPES(distance.type(), "soft_dtw_backward_cuda", ([&] {
        soft_dtw_cuda_backward_kernel<scalar_t><<<blocks,threads>>>(
            distance.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
            R.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
            E.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
            gamma
        );
    }));

    return E;
}