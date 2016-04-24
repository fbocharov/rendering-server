#pragma once

#include <functional>
#include <vector>

#include "nvenc_types.h"

class nvenc_backend;

struct input_buffer
{
    size_t width;
    size_t height;
    CUdeviceptr nv12_device;
    size_t      nv12_stride;
    void *      nvenc_resource; /* After registering cuda memory in nvenc this field stores pointer to it. */
    NV_ENC_INPUT_PTR     input_surface;
    NV_ENC_BUFFER_FORMAT buffer_fmt;
};

struct output_buffer
{
    NV_ENC_OUTPUT_PTR bitstream_buffer;
    size_t bitstream_buffer_size;
    HANDLE output_event;
    std::function<void(void *, size_t)> on_encode_done;
};

struct encode_buffer
{
    output_buffer output = {};
    input_buffer  input  = {};
};

class io_buffer_pool
{
public:
    io_buffer_pool(nvenc_backend & backend, CUcontext & context, size_t size, size_t buffer_width, size_t buffer_height);
    ~io_buffer_pool();

    encode_buffer * get_available();
    encode_buffer * get_pending();

private:
    void destroy();

private:
    std::vector<encode_buffer> buffers_;
    size_t pending_count_;
    size_t available_ix_;
    size_t pending_ix_;

    nvenc_backend & backend_;
    CUcontext & context_;
};
