#pragma once


#include <stdexcept>
#include <string>
#include <cuda.h>
#include "encoder/nvEncodeAPI.h"

struct encoder_exception : public std::runtime_error
{
    encoder_exception(std::string const & what)
        : std::runtime_error(what)
    {}
};

struct cuda_exception : public encoder_exception
{
    cuda_exception(std::string const & what)
        : encoder_exception(what)
    {}
};

struct nvenc_exception : encoder_exception
{
    nvenc_exception(std::string const & what)
        : encoder_exception(what)
    {}
};

#define __cuda_check(expr, on_error)          \
    do {                                      \
        CUresult  ret;                        \
        if ((ret = (expr)) != CUDA_SUCCESS) { \
            on_error;                         \
        }                                     \
    } while (0)

#define __cuda_throw(expr) __cuda_check(expr, throw cuda_exception("Cuda operation " #expr " failed with code " + std::to_string(ret));)


#define __nvenc_check(expr, on_error)           \
    do {                                        \
        NVENCSTATUS ret;                        \
        if ((ret = (expr)) != NV_ENC_SUCCESS) { \
            on_error;                           \
        }                                       \
    } while (0)

#define __nvenc_throw(expr)  __nvenc_check(expr, throw nvenc_exception("nvEnc operation " #expr " failed with code " + std::to_string(ret)))
#define __nvenc_report(expr) __nvenc_check(expr, fprintf(stderr, "nvEnc operation " #expr " failed with code %d\n", ret))
