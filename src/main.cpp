#include <iostream>

#include <GL/glew.h>
#include <GLFW/glfw3.h>

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
    init_graphics();

    opengl_render render({640, 480});
    render.draw(100);
    std::cout << "Hello world!" << std::endl;

    glfwDestroyWindow(window);

    return 0;
}
