#include <windows.h>

#include <cassert>
#include <functional>

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cuda_gl_interop.h>

#include "encoder/nvenc_encoder.h"
#include "encoder/encoder_errors.h"
#include <iostream>

void rgb_to_nv12(cudaArray_t rgb, unsigned char * nv12,
    unsigned int width, unsigned int height, unsigned int nv12_stride);

namespace
{
    void cuda_init(size_t deviceID, CUcontext & context)
    {
        // CUDA interfaces
        __cuda_throw(cuInit(0));

        int  deviceCount = 0;
        __cuda_throw(cuDeviceGetCount(&deviceCount));
        if (deviceID > (unsigned int)deviceCount - 1)
        {
            fprintf(stderr, "Invalid Device Id = %llu\n", deviceID);
            return;
        }

        // Now we get the actual device
        CUdevice  cuda_device = 0;
        __cuda_throw(cuDeviceGet(&cuda_device, deviceID));

        int  SMminor = 0, SMmajor = 0;
        __cuda_throw(cuDeviceComputeCapability(&SMmajor, &SMminor, deviceID));
        if (((SMmajor << 4) + SMminor) < 0x30)
        {
            fprintf(stderr, "GPU %llu does not have NVENC capabilities exiting\n", deviceID);
            return;
        }

        // Create the CUDA context and pop the current one
        CUcontext current_context;
        __cuda_throw(cuCtxCreate(&context, 0, cuda_device));
        __cuda_throw(cuCtxPopCurrent(&current_context));
    }

    void cuda_destroy(CUcontext & context)
    {
        __cuda_throw(cuCtxDestroy(context));
    }
} // anonymous namespace


size_t const nvenc_encoder::CUDA_DEVICE_ID = 0; // main video card
size_t const nvenc_encoder::POOL_SIZE      = 32;

nvenc_encoder::nvenc_encoder(size_t width, size_t height)
    : cuda_context_(std::bind(cuda_init, CUDA_DEVICE_ID, std::placeholders::_1), cuda_destroy)
    , backend_(cuda_context_, width, height)
    , buffer_pool_(backend_, cuda_context_, POOL_SIZE, width, height)
    , eos_event_(backend_.create_async_event())
//    , self_thread_(std::bind(&nvenc_encoder::encode_loop, this))
{}

nvenc_encoder::~nvenc_encoder()
{
//    shutdown();
    flush();
    if (eos_event_)
    {
        backend_.destroy_async_event(eos_event_);
    }
}

void nvenc_encoder::encode(GLuint texture, std::function<void(void *, size_t)> on_done)
{
//    std::lock_guard<std::mutex> lock(guard_);

    encode_buffer * buffer = buffer_pool_.get_available();
    if (!buffer)
    {
        buffer = buffer_pool_.get_pending();
        process_output(buffer);
        if (buffer->input.encoder_input)
        {
            backend_.unmap_input_resource(buffer->input.encoder_input);
            buffer->input.encoder_input = nullptr;
        }
        buffer = buffer_pool_.get_available();
    }

    convert_to_nv12(texture, buffer);

    buffer->input.encoder_input = backend_.map_input_resource(buffer->input.nvenc_resource);
    buffer->output.on_encode_done = on_done;

    std::cerr << "Pushing for encoding fbo " << texture << std::endl;

    backend_.encode_frame(buffer, buffer->input.width, buffer->input.height);
    ready_to_encode_buffers_.push(buffer);
    ready_buffers_.notify_one();
}

void nvenc_encoder::shutdown()
{
    is_interrupted_ = true;
    ready_buffers_.notify_one();
    self_thread_.join();
}

void nvenc_encoder::encode_loop()
{
    std::unique_lock<std::mutex> lock(guard_);

    while (true)
    {
        ready_buffers_.wait(lock, [this] { return is_interrupted_ || !ready_to_encode_buffers_.empty(); });
        if (is_interrupted_)
            break;

        encode_buffer * buffer = ready_to_encode_buffers_.front();
        ready_to_encode_buffers_.pop();

        backend_.encode_frame(buffer, buffer->input.width, buffer->input.height);
    }

    flush();
}

void nvenc_encoder::convert_to_nv12(GLuint rgb_texture, encode_buffer * buffer)
{
    cuda_lock lock(cuda_context_);

    cudaGraphicsResource_t resource;

    cudaError_t ret;
    ret = cudaGraphicsGLRegisterImage(&resource, rgb_texture, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsReadOnly);
    if (ret != cudaSuccess)
    {
        throw cuda_exception(cudaGetErrorString(ret));
    }
    ret = cudaGraphicsMapResources(1, &resource);
    if (ret != cudaSuccess)
    {
        throw cuda_exception(cudaGetErrorString(ret));
    }

    cudaArray_t rgb;
    ret = cudaGraphicsSubResourceGetMappedArray(&rgb, resource, 0, 0);
    if (ret != cudaSuccess)
    {
        throw cuda_exception(cudaGetErrorString(ret));
    }

    rgb_to_nv12(rgb, buffer->input.nv12_device, 
        buffer->input.width, buffer->input.height, buffer->input.nv12_stride);

    cudaGraphicsUnmapResources(1, &resource);
    cudaGraphicsUnregisterResource(resource);
}

void nvenc_encoder::process_output(encode_buffer * buffer)
{
    if (buffer->output.output_event)
    {
        if (WaitForSingleObject(buffer->output.output_event, INFINITE) != WAIT_OBJECT_0)
        {
            assert(0);
        }
    }

    void * bitstream;
    size_t size = backend_.lock_bitstream(buffer->output.bitstream_buffer, &bitstream);
    buffer->output.on_encode_done(bitstream, size);
    backend_.unlock_bitstream(buffer->output.bitstream_buffer);
}

void nvenc_encoder::flush()
{
    backend_.send_eos(eos_event_);

    encode_buffer * buffer = buffer_pool_.get_pending();
    while (buffer)
    {
        process_output(buffer);
        if (buffer && buffer->input.encoder_input)
        {
            backend_.unmap_input_resource(buffer->input.encoder_input);
            buffer->input.encoder_input = nullptr;
        }
        buffer = buffer_pool_.get_pending();
    }

#if defined(NV_WINDOWS)
    if (WaitForSingleObject(eos_event_, 500) != WAIT_OBJECT_0)
    {
        assert(0);
    }
#endif
}
