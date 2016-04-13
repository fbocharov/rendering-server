#pragma once

#include <string>
#include <vector>
#include <fstream>

#include <GL/glew.h>
#include <glm/vec2.hpp>


class opengl_shader
{
public:
    opengl_shader(GLenum type, std::string const & source_path)
        : handler_(glCreateShader(type))
    {
        std::string code = load_file_as_string(source_path);
        char const * code_str = code.c_str();
        glShaderSource(handler_, 1, &code_str, NULL);
        glCompileShader(handler_);
    }

    operator GLuint() const
    {
        return handler_;
    }

    ~opengl_shader()
    {
        glDeleteShader(handler_);
    }

private:
    // need move this to utils?
    std::string load_file_as_string(std::string const & path)
    {
        std::fstream file(path);
        return std::string(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>());
    }

private:
    GLuint handler_;
};


class opengl_program
{
public:
    opengl_program(std::vector<opengl_shader> const & shaders)
        : handler_(glCreateProgram())
    {
        for (auto & shader : shaders)
        {
            glAttachShader(handler_, shader);
        }

        glLinkProgram(handler_);

        for (auto & shader : shaders)
        {
            glDetachShader(handler_, shader);
        }
    }

    operator GLuint()
    {
        return handler_;
    }

    ~opengl_program()
    {
        glDeleteProgram(handler_);
    }

private:
    GLuint handler_;
};

class opengl_vertex_array
{
public:
    opengl_vertex_array()
    {
        glGenVertexArrays(1, &handler_);
    }
    ~opengl_vertex_array()
    {
        glDeleteVertexArrays(1, &handler_);
    }

    operator GLuint()
    {
        return handler_;
    }

private:
    GLuint handler_;
};


class opengl_texture
{
public:
    opengl_texture(glm::uvec2 const & texture_size)
    {
        glActiveTexture(GL_TEXTURE0);
        glGenTextures(1, &handler_);

        glBindTexture(GL_TEXTURE_2D, handler_);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_BASE_LEVEL, 0);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAX_LEVEL, 0);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_SWIZZLE_R, GL_RED);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_SWIZZLE_G, GL_GREEN);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_SWIZZLE_B, GL_BLUE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_SWIZZLE_A, GL_ALPHA);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, GLsizei(texture_size.x), GLsizei(texture_size.y),
            0, GL_RGB, GL_UNSIGNED_BYTE, nullptr);
    }

    ~opengl_texture()
    {
        glDeleteTextures(1, &handler_);
    }

    operator GLuint()
    {
        return handler_;
    }

private:
    GLuint handler_;
};


class opengl_fbo
{
public:
    opengl_fbo()
    {
        glGenFramebuffers(1, &handler_);
    }
    ~opengl_fbo()
    {
        glDeleteFramebuffers(1, &handler_);
    }

    operator GLuint()
    {
        return handler_;
    }

private:
    GLuint handler_;
};
