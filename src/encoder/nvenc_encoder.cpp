#include <windows.h>

#include <cassert>
#include <functional>

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cuda_gl_interop.h>

#include "encoder/nvenc_encoder.h"
#include "encoder/encoder_errors.h"

void rgb_to_nv12(unsigned int * rgb, unsigned char * nv12, unsigned int width, unsigned int height);

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
    // TODO: is it possible to pass placeholder instead of reference param?
    : cuda_context_(std::bind(cuda_init, CUDA_DEVICE_ID, std::placeholders::_1), cuda_destroy)
    , backend_(cuda_context_, width, height)
    , buffer_pool_(backend_, cuda_context_, POOL_SIZE, width, height)
    , eos_event_(backend_.create_async_event())
{
    self_thread_ = std::thread(std::bind(&nvenc_encoder::encode_loop, this));
}

nvenc_encoder::~nvenc_encoder()
{
    shutdown();
    if (eos_event_)
    {
        backend_.destroy_async_event(eos_event_);
    }
}

void nvenc_encoder::encode(GLuint vbo, std::function<void(void *, size_t)> on_done)
{
    std::lock_guard<std::mutex> lock(guard_);

    encode_buffer * buffer = buffer_pool_.get_available();
    if (!buffer)
    {
        buffer = buffer_pool_.get_pending();
        process_output(buffer);
        if (buffer->input.input_surface)
        {
            backend_.unmap_input_resource(buffer->input.input_surface);
            buffer->input.input_surface = nullptr;
        }
        buffer = buffer_pool_.get_available();
    }
    buffer->input.input_surface = backend_.map_input_resource(buffer->input.nvenc_resource);
    buffer->output.on_encode_done = on_done;

    convert_to_nv12(vbo, buffer);
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

void nvenc_encoder::convert_to_nv12(GLuint texture, encode_buffer * buffer)
{
    cuda_lock lock(cuda_context_);

    cudaGraphicsResource_t resource;

    cudaGraphicsGLRegisterImage(&resource, texture, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsReadOnly);
    cudaGraphicsMapResources(1, &resource);

    unsigned int * rgb;
    size_t size;
    size_t width = buffer->input.width;
    size_t height = buffer->input.height;

    cudaGraphicsResourceGetMappedPointer((void **) &rgb, &size, resource);

    rgb_to_nv12(rgb, (unsigned char *) buffer->output.bitstream_buffer, width, height);

    cudaGraphicsUnmapResources(1, &resource);
    cudaGraphicsUnregisterResource(resource);
}

void nvenc_encoder::process_output(encode_buffer * buffer)
{
    backend_.lock_bitstream(buffer->output.bitstream_buffer);
    buffer->output.on_encode_done(buffer->output.bitstream_buffer, buffer->output.bitstream_buffer_size);
    backend_.unlock_bitstream(buffer->output.bitstream_buffer);
}

void nvenc_encoder::flush()
{
    backend_.send_eos(eos_event_);

    encode_buffer * buffer = buffer_pool_.get_pending();
    while (buffer)
    {
        buffer = buffer_pool_.get_pending();
        process_output(buffer);
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
