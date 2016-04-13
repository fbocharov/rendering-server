#pragma once

#include <string>

#include <GL/glew.h>
#include <glm/vec2.hpp>
#include <glm/vec4.hpp>

#include "opengl_wrappers.h"


class opengl_render
{
public:
    explicit opengl_render(glm::uvec2 const & size);

    void draw(size_t frame_count);

private:
    void render();

private:
    opengl_program program_;
    opengl_fbo fbo_;
    opengl_texture texture_;
    opengl_vertex_array array_;
    GLuint shader_diffuse_;
    glm::vec4 viewport_;

private:
    static size_t const TEXTURE_COUNT = 3;
    static std::string const VERTEX_SHADER_SOURCE_PATH;
    static std::string const FRAGMENT_SHADER_SOURCE_PATH;
};
