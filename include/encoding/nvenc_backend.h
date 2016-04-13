#pragma once


#include <memory>
#include <cuda.h>
#include <exception>

#include "nvEncodeAPI.h"

#include "utils.h"
#include "nvenc_types.h"

#if defined (NV_WINDOWS)
#define NVENCAPI __stdcall
#pragma warning(disable : 4996)
#elif defined (NV_UNIX)
#include <dlfcn.h>
#include <string.h>
#define NVENCAPI
#endif

struct encode_buffer;

class nvenc_backend
{
public:
    nvenc_backend(CUcontext cuda_device);
    ~nvenc_backend();

    /// returns registered resource
    void * register_resource(NV_ENC_INPUT_RESOURCE_TYPE type, void * resource, uint32_t width, uint32_t height, uint32_t pitch);
    void unregister_resource(NV_ENC_REGISTERED_PTR resource);
    /// returns created buffer
    void * create_bitstream_buffer(uint32_t size);
    void  destroy_bitstream_buffer(NV_ENC_OUTPUT_PTR bitstream_buffer);
    /// returns created event
    void * create_async_event();
    void  destroy_async_event(void * event);
    ///returns mapped resource
    void * map_input_resource(void * registeredResource);
    void unmap_input_resource(NV_ENC_INPUT_PTR input_resource);

    void encode_frame(encode_buffer * frame, uint32_t width, uint32_t height);

    void send_eos(void * event);

private:
    using api_factory_method = NVENCSTATUS(NVENCAPI *)(NV_ENCODE_API_FUNCTION_LIST*);

private:
    void create_api_instance();
    void open_encode_session(CUcontext cuda_device);
    void init_encoder(size_t width, size_t height);

private:
    scoped_guard<HINSTANCE> library_;
    std::unique_ptr<NV_ENCODE_API_FUNCTION_LIST> api_;
    size_t encode_frames_timestamp_;

    void * encoder_;
};
