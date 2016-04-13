#include <windows.h>

#include <cassert>

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cuda_gl_interop.h>

#include "encoding/nvenc_encoder.h"
#include "encoding/encoder_errors.h"


namespace
{
    void cuda_init(size_t deviceID, CUcontext & context)
    {
        // CUDA interfaces
        __cuda_check(cuInit(0));

        int  deviceCount = 0;
        __cuda_check(cuDeviceGetCount(&deviceCount));
        if (deviceID > (unsigned int)deviceCount - 1)
        {
            fprintf(stderr, "Invalid Device Id = %d\n", deviceID);
            return;
        }

        // Now we get the actual device
        CUdevice  cuda_device = 0;
        __cuda_check(cuDeviceGet(&cuda_device, deviceID));

        int  SMminor = 0, SMmajor = 0;
        __cuda_check(cuDeviceComputeCapability(&SMmajor, &SMminor, deviceID));
        if (((SMmajor << 4) + SMminor) < 0x30)
        {
            fprintf(stderr, "GPU %d does not have NVENC capabilities exiting\n", deviceID);
            return;
        }

        // Create the CUDA context and pop the current one
        CUcontext current_context;
        __cuda_check(cuCtxCreate(&context, 0, cuda_device));
        __cuda_check(cuCtxPopCurrent(&current_context));
    }

    void cuda_destroy(CUcontext & context)
    {
        __cuda_check(cuCtxDestroy(context));
    }
} // anonymous namespace


size_t const nvenc_encoder::CUDA_DEVICE_ID = 0; // main video card
size_t const nvenc_encoder::POOL_SIZE      = 32;

nvenc_encoder::nvenc_encoder()
    // TODO: is it possible to pass placeholder instead of reference param?
    : cuda_context_(std::bind(cuda_init, CUDA_DEVICE_ID, std::placeholders::_1), cuda_destroy)
    , backend_(cuda_context_)
    , buffer_pool_(backend_, cuda_context_, POOL_SIZE, -1, -1) // TODO: insert width and height
    , eos_event_(backend_.create_async_event())
{}

nvenc_encoder::~nvenc_encoder()
{
    flush();
    if (eos_event_)
    {
#if defined(NV_WINDOWS)
        backend_.destroy_async_event(eos_event_);
#endif
    }
}

void nvenc_encoder::encode(GLuint vbo)
{
    encode_buffer * buffer = buffer_pool_.get_available();
    if (!buffer)
    {
        buffer = buffer_pool_.get_pending();
        // TODO: process output (e.g. dump to file)
        // UnMap the input buffer after frame done
        if (buffer->input.input_surface)
        {
            backend_.unmap_input_resource(buffer->input.input_surface);
            buffer->input.input_surface = nullptr;
        }
        buffer = buffer_pool_.get_available();
    }
    buffer->input.input_surface = backend_.map_input_resource(buffer->input.nvenc_resource);

    rgb_to_nv12(vbo, buffer);
    backend_.encode_frame(buffer, -1, -1); // TODO: pass width and height
}

void nvenc_encoder::rgb_to_nv12(GLuint vbo, encode_buffer * buffer)
{
    cudaGraphicsResource_t resource;

    cudaGraphicsGLRegisterBuffer(&resource, vbo, cudaGraphicsRegisterFlagsReadOnly);
    cudaGraphicsMapResources(1, &resource);

    cudaGraphicsUnmapResources(1, &resource);
    cudaGraphicsUnregisterResource(resource);
}

void nvenc_encoder::flush()
{
    backend_.send_eos(eos_event_);

    encode_buffer * buffer = buffer_pool_.get_pending();
    while (buffer)
    {
        buffer = buffer_pool_.get_pending();
        // TODO: process output (e.g. dump to file)
        // UnMap the input buffer after frame is done
        if (buffer && buffer->input.input_surface)
        {
            backend_.unmap_input_resource(buffer->input.input_surface);
            buffer->input.input_surface = nullptr;
        }
    }

#if defined(NV_WINDOWS)
    if (WaitForSingleObject(eos_event_, 500) != WAIT_OBJECT_0)
    {
        assert(0);
    }
#endif
}
