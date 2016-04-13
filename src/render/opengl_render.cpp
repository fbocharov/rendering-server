#include <GL/glew.h>
#include <glm/vec2.hpp>
#include <glm/vec4.hpp>

#include "render/opengl_render.h"


std::string const opengl_render::VERTEX_SHADER_SOURCE_PATH   = "shader/fbo-rtt.vert";
std::string const opengl_render::FRAGMENT_SHADER_SOURCE_PATH = "shader/fbo-rtt.frag";

opengl_render::opengl_render(glm::uvec2 const & size)
    : program_({
        { GL_VERTEX_SHADER,   VERTEX_SHADER_SOURCE_PATH   },
        { GL_FRAGMENT_SHADER, FRAGMENT_SHADER_SOURCE_PATH }
      })
    , texture_(size)
    , shader_diffuse_(glGetUniformLocation(program_, "Diffuse"))
    , viewport_(glm::vec4(0, 0, size))
{
    glBindFramebuffer(GL_FRAMEBUFFER, fbo_);
    glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, texture_, 0);

    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
    {
        throw std::runtime_error("FBO initialization failed.");
    }
}

void opengl_render::draw(size_t frame_count)
{
    for (size_t i = 0; i < frame_count; ++i)
    {
        render();
        // TODO: callback encoder
    }
}

size_t counter = 0;

void opengl_render::render()
{
    glBindFramebuffer(GL_FRAMEBUFFER, fbo_);
    GLenum draw_buffers[1] = { GL_COLOR_ATTACHMENT0 };
    glDrawBuffers(1, draw_buffers);

    if ((counter++ / 50) % 2)
    {
        glClearBufferfv(GL_COLOR, 0, &glm::vec4(0.0f, 0.0f, 1.0f, 1.0f)[0]);
    }
    else
    {
        glClearBufferfv(GL_COLOR, 0, &glm::vec4(1.0f, 1.0f, 0.0f, 0.0f)[0]);
    }

    glUseProgram(program_);
    glUniform1i(shader_diffuse_, 0);
    glBindVertexArray(array_);

    glViewport(GLint(viewport_.x), GLint(viewport_.y),
        GLsizei(viewport_.z), GLsizei(viewport_.w));
}
