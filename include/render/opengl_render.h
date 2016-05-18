#pragma once

#include <string>
#include <functional>

#include <GL/glew.h>
#include <glm/vec2.hpp>
#include <glm/vec4.hpp>


class opengl_render
{
public:
    explicit opengl_render(glm::uvec2 const & size);
    ~opengl_render();

    void draw(size_t frame_count, std::function<void(GLuint)> on_render);

private:
    void render();

private:
    GLuint vaoCube, vaoQuad;
    GLuint vboCube, vboQuad;
    GLuint sceneVertexShader, sceneFragmentShader, sceneShaderProgram;
    GLuint screenVertexShader, screenFragmentShader, screenShaderProgram;
    GLuint texKitten, texPuppy;
    GLint uniModel;
    GLuint frameBuffer;
    GLuint texColorBuffer;
    GLuint rboDepthStencil;
    GLint uniView;
    GLint uniProj;
    GLint uniColor;
    std::chrono::time_point<std::chrono::high_resolution_clock> t_start;
};
