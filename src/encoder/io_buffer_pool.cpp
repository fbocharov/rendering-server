#include <cuda.h>

#include "encoder/nvCPUOPSys.h"
#include "encoder/io_buffer_pool.h"
#include "encoder/nvenc_backend.h"
#include "encoder/encoder_errors.h"
#include "utils.h"

#define BITSTREAM_BUFFER_SIZE 2 * 1024 * 1024

io_buffer_pool::io_buffer_pool(nvenc_backend & backend, CUcontext & context, size_t size, size_t buffer_width, size_t buffer_height)
    : backend_(backend)
    , context_(context)
{
    cuda_lock lock(context_);

    try {
        for (size_t i = 0; i < size; ++i)
        {
            encode_buffer buffer = {};
            __cuda_throw(cuMemAllocPitch(&buffer.input.nv12_device, (size_t *)&buffer.input.nv12_stride, buffer_width, buffer_height * 3 / 2, 16));

            buffer.input.nvenc_resource = backend_.register_resource(NV_ENC_INPUT_RESOURCE_TYPE_CUDADEVICEPTR, (void *)buffer.input.nv12_device,
                buffer_width, buffer_height, buffer.input.nv12_stride);

            buffer.input.buffer_fmt = NV_ENC_BUFFER_FORMAT_NV12_PL;
            buffer.input.width = buffer_width;
            buffer.input.height = buffer_height;

            buffer.output.bitstream_buffer = backend_.create_bitstream_buffer(BITSTREAM_BUFFER_SIZE);
            buffer.output.bitstream_buffer_size = BITSTREAM_BUFFER_SIZE;

#if defined(NV_WINDOWS)
            buffer.output.output_event = backend_.create_async_event();
#else
            buffer.output.output_event = nullptr;
#endif
            buffers_.emplace_back(buffer);
        }
    }
    catch (nvenc_exception &)
    {
        destroy();
        throw;
    }
}

io_buffer_pool::~io_buffer_pool()
{
    cuda_lock lock(context_);
    destroy();
}

void io_buffer_pool::destroy()
{
    for (auto it = buffers_.rbegin(); it != buffers_.rend(); ++it)
    {
        if (it->input.nvenc_resource)
        {
            backend_.unregister_resource(it->input.nvenc_resource);
        }

        if (it->input.nv12_device)
        {
            cuMemFree(it->input.nv12_device);
        }

        if (it->output.bitstream_buffer)
        {
            backend_.destroy_bitstream_buffer(it->output.bitstream_buffer);
        }

#if defined(NV_WINDOWS)
        if (it->output.output_event)
        {
            backend_.destroy_async_event(it->output.output_event);
        }
#endif
    }
}


encode_buffer * io_buffer_pool::get_available()
{
    if (pending_count_ == buffers_.size())
    {
        return nullptr;
    }

    encode_buffer * buffer = &buffers_[available_ix_];
    available_ix_ = (available_ix_ + 1) % buffers_.size();
    pending_count_ += 1;

    return buffer;
}

encode_buffer * io_buffer_pool::get_pending()
{
    if (pending_count_ == 0)
    {
        return NULL;
    }

    encode_buffer * buffer = &buffers_[available_ix_];
    available_ix_ = (available_ix_ + 1) % buffers_.size();
    available_ix_ -= 1;

    return buffer;
}
