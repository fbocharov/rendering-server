// About NV12 format you can read here: http://www.fourcc.org/yuv.php#NV12

extern "C" __global__ 
void __rgb_to_nv12(unsigned int * rgb, unsigned char * nv12, unsigned int width, unsigned int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned char * Y    = (unsigned char *) nv12;
    unsigned char * CbCr = (unsigned char *) (nv12 + width * height);

    if ((x < width) && (y < height))
    {
        unsigned char * pixel = (unsigned char *) (rgb + y * width + x);
        unsigned int r = pixel[0];
        unsigned int g = pixel[1];
        unsigned int b = pixel[2];

        // Convertion formulas taken from https://en.wikipedia.org/wiki/YUV#Converting_between_Y.27UV_and_RGB
        Y[y * width + x] = 0.299 * r + 0.587 * g + 0.114 * b;
        if (x % 2 == 0 && y % 2 == 0)
        {
            y /= 2;
            CbCr[y  * width + x]     = -0.169 * r - 0.331 * g + 0.499  * b + 128;
            CbCr[y * width + x + 1] =  0.499 * r - 0.418 * g - 0.0813 * b + 128;
        }
    }
}

void rgb_to_nv12(unsigned int * rgb, unsigned char * nv12, unsigned int width, unsigned int height) 
{
    dim3 dim_block(16, 16, 1);
    dim3 dim_grid(width / dim_block.x, height / dim_block.y, 1);
    __rgb_to_nv12<<<dim_block, dim_grid>>>(rgb, nv12, width, height);
}