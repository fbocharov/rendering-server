// About NV12 format you can read here: http://www.fourcc.org/yuv.php#NV12
#include "encoder/encoder_errors.h"

extern "C" __global__
void __rgb_to_nv12(unsigned int * rgb, unsigned char * nv12,
    unsigned int width, unsigned int height, unsigned int nv12_stride)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned char * Y    = nv12;
    unsigned char * CbCr = nv12 + nv12_stride * height;

    if ((j < width) && (i < height))
    {
        unsigned char * pixel = (unsigned char *) (rgb + i * width + j);
        unsigned int r = pixel[0];
        unsigned int g = pixel[1];
        unsigned int b = pixel[2];

        // Convertion formulas taken from http://www.fourcc.org/fccyvrgb.php
        Y[i * nv12_stride + j] = 0.257 * r + 0.504 * g + 0.098 * b + 16;
        if (i % 2 == 0 && j % 2 == 0)
        {
            i /= 2;
            CbCr[i * nv12_stride + j]     =  0.439 * r - 0.368 * g - 0.071 * b + 128;
            CbCr[i * nv12_stride + j + 1] = -0.148 * r - 0.291 * g + 0.439 * b + 128;
        }
    }
}

void rgb_to_nv12(unsigned int * rgb, unsigned char * nv12,
    unsigned int width, unsigned int height, unsigned int nv12_stride)
{
    dim3 threads_per_block(16, 16);
    dim3 block_count(
        (width  + threads_per_block.x - 1) / threads_per_block.x,
        (height + threads_per_block.y - 1) / threads_per_block.y
    );

    __rgb_to_nv12<<<block_count, threads_per_block>>>(rgb, nv12, width, height, nv12_stride);
    cudaDeviceSynchronize();
    cudaError_t ret = cudaGetLastError();
    if (ret != cudaSuccess)
    {
        throw cuda_exception(cudaGetErrorString(ret));
    }
}
