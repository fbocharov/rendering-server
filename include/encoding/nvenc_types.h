#pragma once

#include <cstdio>

#include "nvCPUOPSys.h"
#include "nvEncodeAPI.h"

#if defined (NV_WINDOWS)
#include "d3d9.h"
#define NVENCAPI __stdcall
#pragma warning(disable : 4996)
#elif defined (NV_UNIX)
#include <dlfcn.h>
#include <string.h>
#define NVENCAPI
#endif

struct EncodeFrameConfig
{
    uint8_t  *yuv[3];
    uint32_t stride[3];
    uint32_t width;
    uint32_t height;
};
