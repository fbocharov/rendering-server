#include <stdexcept>

#include <cuda.h>

#include "encoder/nvCPUOPSys.h"

#include "encoder/nvenc_backend.h"
#include "encoder/io_buffer_pool.h"
#include "encoder/encoder_errors.h"


#define SET_VER(configStruct, type) {configStruct.version = type##_VER;}

namespace
{
    void load_library(HINSTANCE & library)
    {
#if defined(NV_WINDOWS)
#if defined (_WIN64)
        library = LoadLibrary(TEXT("nvEncodeAPI64.dll"));
#else
        library = LoadLibrary(TEXT("nvEncodeAPI.dll"));
#endif
#else
        library = dlopen("libnvidia-encode.so.1", RTLD_LAZY);
#endif
        if (!library)
            throw nvenc_exception("Can't load NVENC library.");
    }

    void unload_library(HINSTANCE & library)
    {
        if (library)
        {
#if defined (NV_WINDOWS)
            FreeLibrary(library);
#else
            dlclose(library);
#endif
        }
    }
}  // anonymous namespace


nvenc_backend::nvenc_backend(CUcontext cuda_device, size_t width, size_t height)
    : library_(load_library, unload_library)
{
    create_api_instance();
    open_encode_session(cuda_device);
    init_encoder(width, height);
}

nvenc_backend::~nvenc_backend()
{
    api_->nvEncDestroyEncoder(encoder_);
}

void * nvenc_backend::register_resource(NV_ENC_INPUT_RESOURCE_TYPE type, void * resource, uint32_t width, uint32_t height, uint32_t pitch)
{
    NV_ENC_REGISTER_RESOURCE params = {};

    SET_VER(params, NV_ENC_REGISTER_RESOURCE);
    params.resourceType = type;
    params.resourceToRegister = resource;
    params.width = width;
    params.height = height;
    params.pitch = pitch;
    params.bufferFormat = NV_ENC_BUFFER_FORMAT_NV12_PL;

    __nvenc_throw(api_->nvEncRegisterResource(encoder_, &params));

    return params.registeredResource;
}

void nvenc_backend::unregister_resource(NV_ENC_REGISTERED_PTR resource)
{
    __nvenc_throw(api_->nvEncUnregisterResource(encoder_, resource));
}


void * nvenc_backend::create_bitstream_buffer(uint32_t size)
{
    NV_ENC_CREATE_BITSTREAM_BUFFER params = {};

    SET_VER(params, NV_ENC_CREATE_BITSTREAM_BUFFER);
    params.size = size;
    params.memoryHeap = NV_ENC_MEMORY_HEAP_SYSMEM_CACHED;

    __nvenc_throw(api_->nvEncCreateBitstreamBuffer(encoder_, &params));

    return params.bitstreamBuffer;
}

void nvenc_backend::destroy_bitstream_buffer(NV_ENC_OUTPUT_PTR bitstream_buffer)
{
    if (bitstream_buffer)
    {
        __nvenc_throw(api_->nvEncDestroyBitstreamBuffer(encoder_, bitstream_buffer));
    }
}

void nvenc_backend::lock_bitstream(void * bitstream)
{
    NV_ENC_LOCK_BITSTREAM params = {};
    SET_VER(params, NV_ENC_LOCK_BITSTREAM);
    params.outputBitstream = bitstream;
    params.doNotWait = false;
    __nvenc_throw(api_->nvEncLockBitstream(encoder_, &params));
}

void nvenc_backend::unlock_bitstream(void * bitstream)
{
    __nvenc_throw(api_->nvEncUnlockBitstream(encoder_, bitstream));
}

void * nvenc_backend::create_async_event()
{
    NV_ENC_EVENT_PARAMS params = {};

    SET_VER(params, NV_ENC_EVENT_PARAMS);
#if defined (NV_WINDOWS)
    params.completionEvent = CreateEvent(nullptr, FALSE, FALSE, nullptr);
#else
    eventParams.completionEvent = NULL;
#endif
    __nvenc_throw(api_->nvEncRegisterAsyncEvent(encoder_, &params));

    return params.completionEvent;
}

void nvenc_backend::destroy_async_event(void * event)
{
    if (event)
    {
        NV_ENC_EVENT_PARAMS params = {};

        SET_VER(params, NV_ENC_EVENT_PARAMS);
        params.completionEvent = event;

        __nvenc_throw(api_->nvEncUnregisterAsyncEvent(encoder_, &params));

#if defined (NV_WINDOWS)
        CloseHandle(event);
#endif
    }
}


void * nvenc_backend::map_input_resource(void * input_resource)
{
    NV_ENC_MAP_INPUT_RESOURCE params = {};

    SET_VER(params, NV_ENC_MAP_INPUT_RESOURCE);
    params.registeredResource = input_resource;

    __nvenc_throw(api_->nvEncMapInputResource(encoder_, &params));

    return params.mappedResource;
}

void nvenc_backend::unmap_input_resource(NV_ENC_INPUT_PTR input_resource)
{
    if (input_resource)
    {
        __nvenc_throw(api_->nvEncUnmapInputResource(encoder_, input_resource));
    }
}

void nvenc_backend::send_eos(void * event)
{
    NV_ENC_PIC_PARAMS encPicParams = {};

    SET_VER(encPicParams, NV_ENC_PIC_PARAMS);
    encPicParams.encodePicFlags = NV_ENC_PIC_FLAG_EOS;
    encPicParams.completionEvent = event;

    __nvenc_throw(api_->nvEncEncodePicture(encoder_, &encPicParams));
}

void nvenc_backend::encode_frame(encode_buffer * frame, uint32_t width, uint32_t height)
{
    NV_ENC_PIC_PARAMS params = {};

    SET_VER(params, NV_ENC_PIC_PARAMS);
    params.inputBuffer = frame->input.input_surface;
    params.bufferFmt = frame->input.buffer_fmt;
    params.inputWidth = width;
    params.inputHeight = height;
    params.outputBitstream = frame->input.input_surface;
    params.completionEvent = frame->output.output_event;
    params.inputTimeStamp = encode_frames_timestamp_;

    NVENCSTATUS nvStatus = api_->nvEncEncodePicture(encoder_, &params);
    if (nvStatus != NV_ENC_SUCCESS && nvStatus != NV_ENC_ERR_NEED_MORE_INPUT)
    {
        throw nvenc_exception("Can't encode frame.");
    }

    ++encode_frames_timestamp_;
}

void nvenc_backend::create_api_instance()
{
#if defined(NV_WINDOWS)
    auto make_api_instance = reinterpret_cast<api_factory_method>(GetProcAddress(library_, "NvEncodeAPICreateInstance"));
#else
    nvEncodeAPICreateInstance = (api_factory_method)dlsym(lib_instance_, "NvEncodeAPICreateInstance");
#endif

    if (!make_api_instance)
        throw nvenc_exception("Can't load create instance method from NVENC dll.");

    api_.reset(new NV_ENCODE_API_FUNCTION_LIST{});
    api_->version = NV_ENCODE_API_FUNCTION_LIST_VER;
    if (make_api_instance(api_.get()))
        throw nvenc_exception("Can't create NVENC api instance.");
}

void nvenc_backend::open_encode_session(CUcontext cuda_device)
{
    NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS params = {};
    params.device = cuda_device;
    params.deviceType = NV_ENC_DEVICE_TYPE_CUDA;
    params.apiVersion = NVENCAPI_VERSION;

    if (NV_ENC_SUCCESS != api_->nvEncOpenEncodeSessionEx(&params ,&encoder_))
    {
        api_->nvEncDestroyEncoder(encoder_);
        throw nvenc_exception("Can't open encode session.");
    }
}

// NOTE: if you want explanation of all magic constants there
// see structure description in header file.
// Also Nvidia guide recommends some presets for low latency encoding.
// See it there: http://goo.gl/TGBDqd
void nvenc_backend::init_encoder(size_t width, size_t height)
{
    NV_ENC_INITIALIZE_PARAMS params = {};
    SET_VER(params, NV_ENC_INITIALIZE_PARAMS);

    params.encodeGUID = NV_ENC_CODEC_H264_GUID;
    params.presetGUID = NV_ENC_PRESET_LOW_LATENCY_HQ_GUID;
    params.encodeWidth = width;
    params.encodeHeight = height;

    params.darWidth = width;
    params.darHeight = height;
    params.frameRateNum = 30; // Set desired FPS there
    params.frameRateDen = 1;
#if defined(NV_WINDOWS)
    params.enableEncodeAsync = 1;
#else
    m_stCreateEncodeParams.enableEncodeAsync = 0;
#endif
    params.enablePTD = 1;
    params.maxEncodeWidth = width;
    params.maxEncodeHeight = height;

    NV_ENC_PRESET_CONFIG preset_config = {};
    SET_VER(preset_config, NV_ENC_PRESET_CONFIG);
    SET_VER(preset_config.presetCfg, NV_ENC_CONFIG);
    __nvenc_throw(api_->nvEncGetEncodePresetConfig(encoder_, params.encodeGUID, params.presetGUID, &preset_config));

    NV_ENC_CONFIG encode_config = {};
    SET_VER(encode_config, NV_ENC_CONFIG);
    memcpy(&encode_config, &preset_config.presetCfg, sizeof(NV_ENC_CONFIG));
    params.encodeConfig = &encode_config;

    encode_config.gopLength = NVENC_INFINITE_GOPLENGTH;
    encode_config.frameIntervalP = 1;
    encode_config.frameFieldMode = NV_ENC_PARAMS_FRAME_FIELD_MODE_FRAME;

    encode_config.mvPrecision = NV_ENC_MV_PRECISION_QUARTER_PEL;

    encode_config.rcParams.enableAQ = 1;
    encode_config.rcParams.rateControlMode = NV_ENC_PARAMS_RC_2_PASS_QUALITY;
    encode_config.rcParams.averageBitRate = 5000000; // set desired bitrate there

    encode_config.encodeCodecConfig.h264Config.chromaFormatIDC = 1;
    encode_config.encodeCodecConfig.h264Config.idrPeriod = NVENC_INFINITE_GOPLENGTH;

    __nvenc_throw(api_->nvEncInitializeEncoder(encoder_, &params));
}
