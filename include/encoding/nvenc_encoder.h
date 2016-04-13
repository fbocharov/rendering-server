#include <GL/gl.h>

#include "nvenc_backend.h"
#include "io_buffer_pool.h"
#include "utils.h"


class nvenc_encoder
{
public:
    explicit nvenc_encoder();
    ~nvenc_encoder();

    void encode(GLuint vbo);

private:
    void rgb_to_nv12(GLuint vbo, encode_buffer * buffer);
    void flush();

private:
    // NOTE: field initialization order is important!
    scoped_guard<CUcontext> cuda_context_;
    nvenc_backend backend_;
    io_buffer_pool buffer_pool_;
    void * eos_event_;

    static size_t const CUDA_DEVICE_ID;
    static size_t const POOL_SIZE;
};
