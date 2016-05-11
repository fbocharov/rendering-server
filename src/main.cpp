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

    window = glfwCreateWindow(640, 480, "hello world", nullptr, nullptr);
    glfwMakeContextCurrent(window);

    glewExperimental = GL_TRUE;
    if (GLEW_OK != glewInit())
        throw std::runtime_error("Can't initialize GLEW.");
    glGetError();
}


int main(int argc, char * argv[])
{
    using namespace std::placeholders;

    init_graphics();

    FILE * file = fopen("out.mp4", "wb");
    int frame = 0;

    auto dump_to_file = [&file, &frame] (void * data, size_t size)
    {
        std::cout << "CALLBACK: writing " << size << " bytes in " << frame++ << " frame" << std::endl;
        fwrite(data, 1, size, file);
        fflush(file);
    };


    size_t width = 720, height = 480;
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
