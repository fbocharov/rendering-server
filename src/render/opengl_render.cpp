#include <GL/glew.h>
#include <glm/vec2.hpp>
#include <glm/vec4.hpp>

#include "render/opengl_render.h"


std::string const opengl_render::VERTEX_SHADER =
"#version 400 core\n"
"precision highp float;\n"
"precision highp int;\n"

"layout(std140, column_major) uniform;\n"

"const int VertexCount = 3;\n"
"const vec2 Position[VertexCount] = vec2[](\n"
"	vec2(-1.0,-1.0),\n"
"	vec2( 3.0,-1.0),\n"
"	vec2(-1.0, 3.0));\n"

"void main() {\n"	
"	gl_Position = vec4(Position[gl_VertexID], 0.0, 1.0);\n"
"}\n";

std::string const opengl_render::FRAGMENT_SHADER =
"#version 400 core\n"
"#define FRAG_COLOR 0\n"

"precision highp float;\n"
"precision highp int;\n"
"layout(std140, column_major) uniform;\n"

"uniform sampler2D Diffuse;\n"

"in vec4 gl_FragCoord;\n"
"layout(location = FRAG_COLOR, index = 0) out vec4 Color;\n"

"void main()\n"
"{\n"
"    vec2 TextureSize = vec2(textureSize(Diffuse, 0));\n"
"    Color = texture(Diffuse, gl_FragCoord.xy / TextureSize);\n"
"}";


opengl_render::opengl_render(glm::uvec2 const & size)
    : program_({
        { GL_VERTEX_SHADER,   VERTEX_SHADER   },
        { GL_FRAGMENT_SHADER, FRAGMENT_SHADER }
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
