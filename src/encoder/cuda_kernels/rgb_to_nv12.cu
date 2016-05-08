// About NV12 format you can read here: http://www.fourcc.org/yuv.php#NV12
#include "encoder/encoder_errors.h"

extern "C" __global__ 
void __rgb_to_nv12(unsigned int * rgb, unsigned char * nv12, 
    unsigned int width, unsigned int height, unsigned int nv12_stride)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned char * Y    = (unsigned char *) nv12;
    unsigned char * CbCr = (unsigned char *) (nv12 + nv12_stride * height);

    if ((x < width) && (y < height))
    {
        unsigned char * pixel = (unsigned char *) (rgb + y * width + x);
        unsigned int r = pixel[0];
        unsigned int g = pixel[1];
        unsigned int b = pixel[2];

        // Convertion formulas taken from https://en.wikipedia.org/wiki/YUV#Converting_between_Y.27UV_and_RGB
        Y[y * nv12_stride + x] = 0.299 * r + 0.587 * g + 0.114 * b;
        if (x % 2 == 0 && y % 2 == 0)
        {
            y /= 2;
            CbCr[y * nv12_stride + x]     = -0.169 * r - 0.331 * g + 0.499  * b + 128;
            CbCr[y * nv12_stride + x + 1] =  0.499 * r - 0.418 * g - 0.0813 * b + 128;
        }
    }
}

void rgb_to_nv12(unsigned int * rgb, unsigned char * nv12, 
    unsigned int width, unsigned int height, unsigned int nv12_stride)
{
    dim3 dim_block(16, 16, 1);
    dim3 dim_grid((width + 16 - 1) / 16, (height + 16 - 1) / 16, 1);
    __rgb_to_nv12<<<dim_grid, dim_block>>>(rgb, nv12, width, height, nv12_stride);

    cudaError_t ret = cudaGetLastError();
    if (ret != cudaSuccess)
    {
        throw cuda_exception(cudaGetErrorString(ret));
    }
}
