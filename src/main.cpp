#include <iostream>
#include <functional>

#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include "encoder/nvenc_encoder.h"
#include "encoder/encoder_errors.h"

#include "render/opengl_render.h"

GLFWwindow * window;

void init_graphics()
{
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
    glfwWindowHint(GLFW_CLIENT_API, GLFW_OPENGL_API);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
    glfwWindowHint(GLFW_OPENGL_DEBUG_CONTEXT, GL_TRUE);

    window = glfwCreateWindow(800, 600, "hello world", nullptr, nullptr);
    glfwMakeContextCurrent(window);

    glewExperimental = GL_TRUE;
    if (GLEW_OK != glewInit())
        throw std::runtime_error("Can't initialize GLEW.");
    glGetError();
}


size_t get_nalu_size(char const * frame, size_t size)
{
    for (size_t i = 4; i + 3 < size; ++i)
        if (frame[i] == frame[i + 1] == frame[i + 2] == 0x0 && frame[i + 3] == 0x1)
            return i;

    return size;
}

int main(int argc, char * argv[])
{
    using namespace std::placeholders;

    init_graphics();

    FILE * file = fopen("outtt.mp4", "wb");
    int frame = 0;

    auto dump_to_file = [&file, &frame] (void * data, size_t size)
    {
        size_t nalu_size = get_nalu_size((char const *)data, size);
        fwrite(data, 1, nalu_size, file);
        fwrite((char *)data + nalu_size, 1, size - nalu_size, file);
        std::cout << "CALLBACK: writing " << size << " bytes in " << frame++ << " frame" << std::endl;
        fflush(file);
    };

    size_t width = 800;
    size_t height = 600;
    try
    {
        opengl_render render({width, height});
        nvenc_encoder encoder(width, height);

        auto on_render = std::bind(&nvenc_encoder::encode, std::ref(encoder), _1, dump_to_file);
        render.draw(500, on_render);
    }
    catch (encoder_exception & e)
    {
        std::cerr << e.what() << std::endl;
    }

    glfwDestroyWindow(window);
    fclose(file);

    return 0;
}
