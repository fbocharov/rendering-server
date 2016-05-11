// About NV12 format you can read here: http://www.fourcc.org/yuv.php#NV12
#include "encoder/encoder_errors.h"

namespace {
    texture<uchar4, cudaTextureType2D, cudaReadModeElementType> rgb_texture;
}

extern "C" __global__
void __rgb_to_nv12(unsigned char * nv12,
    unsigned int width, unsigned int height, unsigned int nv12_stride)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned char * Y    = nv12;
    unsigned char * CbCr = nv12 + nv12_stride * height;

    if ((i < width) && (j < height))
    {
        uchar4 pixel = tex2D(rgb_texture, i, j);
        unsigned int b = pixel.x;
        unsigned int g = pixel.y;
        unsigned int r = pixel.z;

        // Convertion formulas taken from http://www.fourcc.org/fccyvrgb.php
        Y[j * nv12_stride + i] = 0.257 * r + 0.504 * g + 0.098 * b + 16;
        if (i % 2 == 0 && j % 2 == 0)
        {
            j /= 2;
            CbCr[j * nv12_stride + i]     =  0.439 * r - 0.368 * g - 0.071 * b + 128;
            CbCr[j * nv12_stride + i + 1] = -0.148 * r - 0.291 * g + 0.439 * b + 128;
        }
    }
}

void rgb_to_nv12(cudaArray_t rgb, unsigned char * nv12,
    unsigned int width, unsigned int height, unsigned int nv12_stride)
{
    dim3 threads_per_block(16, 16);
    dim3 block_count(
        (width  + threads_per_block.x - 1) / threads_per_block.x,
        (height + threads_per_block.y - 1) / threads_per_block.y
    );

    cudaError_t ret = cudaBindTextureToArray(rgb_texture, rgb);
    if (ret != cudaSuccess)
    {
        throw cuda_exception(cudaGetErrorString(ret));
    }

    __rgb_to_nv12<<<block_count, threads_per_block>>>(nv12, width, height, nv12_stride);
    ret = cudaDeviceSynchronize();
    if (ret != cudaSuccess)
    {
        throw cuda_exception(cudaGetErrorString(ret));
    }
}
