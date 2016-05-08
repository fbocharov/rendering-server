#include <thread>
#include <atomic>
#include <mutex>
#include <queue>
#include <condition_variable>

#include <GL/gl.h>

#include "encoder/nvenc_backend.h"
#include "encoder/io_buffer_pool.h"
#include "utils.h"


class nvenc_encoder
{
public:
    explicit nvenc_encoder(size_t width, size_t height);
    ~nvenc_encoder();

    void encode(GLuint texture, std::function<void(void *, size_t)> on_done);

private:
    void encode_loop();
    void convert_to_nv12(GLuint vbo, encode_buffer * buffer);
    void process_output(encode_buffer * buffer);
    void shutdown();
    void flush();

private:
    // NOTE: field order is important for right initialization!
    scoped_guard<CUcontext> cuda_context_;
    nvenc_backend backend_;
    io_buffer_pool buffer_pool_;
    void * eos_event_;

    std::condition_variable ready_buffers_;
    std::queue<encode_buffer *> ready_to_encode_buffers_;

    std::atomic_bool is_interrupted_;
    std::mutex guard_;
    std::thread self_thread_;

    static size_t const CUDA_DEVICE_ID;
    static size_t const POOL_SIZE;
};
