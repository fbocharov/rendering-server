#ifndef GLEW_STATIC
    #define GLEW_STATIC
#endif  // GLEW_STATIC

#include <chrono>

#include <GL/glew.h>
#include <glm/vec2.hpp>
#include <glm/vec4.hpp>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <SOIL.h>

#include "render/opengl_render.h"


// Shader sources
const GLchar* sceneVertexSource =
"#version 150 core\n"
"in vec3 position;"
"in vec3 color;"
"in vec2 texcoord;"
"out vec3 Color;"
"out vec2 Texcoord;"
"uniform mat4 model;"
"uniform mat4 view;"
"uniform mat4 proj;"
"uniform vec3 overrideColor;"
"void main()"
"{"
"    Color = overrideColor * color;"
"    Texcoord = texcoord;"
"    gl_Position = proj * view * model * vec4(position, 1.0);"
"}";
const GLchar* sceneFragmentSource =
"#version 150 core\n"
"in vec3 Color;"
"in vec2 Texcoord;"
"out vec4 outColor;"
"uniform sampler2D texKitten;"
"uniform sampler2D texPuppy;"
"void main()"
"{"
"    outColor = vec4(Color, 1.0) * mix(texture(texKitten, Texcoord), texture(texPuppy, Texcoord), 0.5);"
"}";

const GLchar* screenVertexSource =
"#version 150 core\n"
"in vec2 position;"
"in vec2 texcoord;"
"out vec2 Texcoord;"
"void main()"
"{"
"    Texcoord = texcoord;"
"    gl_Position = vec4(position, 0.0, 1.0);"
"}";
const GLchar* screenFragmentSource =
"#version 150 core\n"
"in vec2 Texcoord;"
"out vec4 outColor;"
"uniform sampler2D texFramebuffer;"
"void main()"
"{"
"    outColor = texture(texFramebuffer, Texcoord);"
"}";

// Cube vertices
GLfloat cubeVertices[] = {
    -0.5f, -0.5f, -0.5f, 1.0f, 1.0f, 1.0f, 0.0f, 0.0f,
    0.5f, -0.5f, -0.5f, 1.0f, 1.0f, 1.0f, 1.0f, 0.0f,
    0.5f,  0.5f, -0.5f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
    0.5f,  0.5f, -0.5f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
    -0.5f,  0.5f, -0.5f, 1.0f, 1.0f, 1.0f, 0.0f, 1.0f,
    -0.5f, -0.5f, -0.5f, 1.0f, 1.0f, 1.0f, 0.0f, 0.0f,

    -0.5f, -0.5f,  0.5f, 1.0f, 1.0f, 1.0f, 0.0f, 0.0f,
    0.5f, -0.5f,  0.5f, 1.0f, 1.0f, 1.0f, 1.0f, 0.0f,
    0.5f,  0.5f,  0.5f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
    0.5f,  0.5f,  0.5f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
    -0.5f,  0.5f,  0.5f, 1.0f, 1.0f, 1.0f, 0.0f, 1.0f,
    -0.5f, -0.5f,  0.5f, 1.0f, 1.0f, 1.0f, 0.0f, 0.0f,

    -0.5f,  0.5f,  0.5f, 1.0f, 1.0f, 1.0f, 1.0f, 0.0f,
    -0.5f,  0.5f, -0.5f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
    -0.5f, -0.5f, -0.5f, 1.0f, 1.0f, 1.0f, 0.0f, 1.0f,
    -0.5f, -0.5f, -0.5f, 1.0f, 1.0f, 1.0f, 0.0f, 1.0f,
    -0.5f, -0.5f,  0.5f, 1.0f, 1.0f, 1.0f, 0.0f, 0.0f,
    -0.5f,  0.5f,  0.5f, 1.0f, 1.0f, 1.0f, 1.0f, 0.0f,

    0.5f,  0.5f,  0.5f, 1.0f, 1.0f, 1.0f, 1.0f, 0.0f,
    0.5f,  0.5f, -0.5f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
    0.5f, -0.5f, -0.5f, 1.0f, 1.0f, 1.0f, 0.0f, 1.0f,
    0.5f, -0.5f, -0.5f, 1.0f, 1.0f, 1.0f, 0.0f, 1.0f,
    0.5f, -0.5f,  0.5f, 1.0f, 1.0f, 1.0f, 0.0f, 0.0f,
    0.5f,  0.5f,  0.5f, 1.0f, 1.0f, 1.0f, 1.0f, 0.0f,

    -0.5f, -0.5f, -0.5f, 1.0f, 1.0f, 1.0f, 0.0f, 1.0f,
    0.5f, -0.5f, -0.5f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
    0.5f, -0.5f,  0.5f, 1.0f, 1.0f, 1.0f, 1.0f, 0.0f,
    0.5f, -0.5f,  0.5f, 1.0f, 1.0f, 1.0f, 1.0f, 0.0f,
    -0.5f, -0.5f,  0.5f, 1.0f, 1.0f, 1.0f, 0.0f, 0.0f,
    -0.5f, -0.5f, -0.5f, 1.0f, 1.0f, 1.0f, 0.0f, 1.0f,

    -0.5f,  0.5f, -0.5f, 1.0f, 1.0f, 1.0f, 0.0f, 1.0f,
    0.5f,  0.5f, -0.5f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
    0.5f,  0.5f,  0.5f, 1.0f, 1.0f, 1.0f, 1.0f, 0.0f,
    0.5f,  0.5f,  0.5f, 1.0f, 1.0f, 1.0f, 1.0f, 0.0f,
    -0.5f,  0.5f,  0.5f, 1.0f, 1.0f, 1.0f, 0.0f, 0.0f,
    -0.5f,  0.5f, -0.5f, 1.0f, 1.0f, 1.0f, 0.0f, 1.0f,

    -1.0f, -1.0f, -0.5f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
    1.0f, -1.0f, -0.5f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f,
    1.0f,  1.0f, -0.5f, 0.0f, 0.0f, 0.0f, 1.0f, 1.0f,
    1.0f,  1.0f, -0.5f, 0.0f, 0.0f, 0.0f, 1.0f, 1.0f,
    -1.0f,  1.0f, -0.5f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f,
    -1.0f, -1.0f, -0.5f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f
};

// Quad vertices
GLfloat quadVertices[] = {
    -1.0f,  1.0f,  0.0f, 1.0f,
    1.0f,  1.0f,  1.0f, 1.0f,
    1.0f, -1.0f,  1.0f, 0.0f,

    1.0f, -1.0f,  1.0f, 0.0f,
    -1.0f, -1.0f,  0.0f, 0.0f,
    -1.0f,  1.0f,  0.0f, 1.0f
};


void createShaderProgram(const GLchar* vertSrc, const GLchar* fragSrc, GLuint& vertexShader, GLuint& fragmentShader, GLuint& shaderProgram)
{
    // Create and compile the vertex shader
    vertexShader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertexShader, 1, &vertSrc, NULL);
    glCompileShader(vertexShader);

    // Create and compile the fragment shader
    fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragmentShader, 1, &fragSrc, NULL);
    glCompileShader(fragmentShader);

    // Link the vertex and fragment shader into a shader program
    shaderProgram = glCreateProgram();
    glAttachShader(shaderProgram, vertexShader);
    glAttachShader(shaderProgram, fragmentShader);
    glBindFragDataLocation(shaderProgram, 0, "outColor");
    glLinkProgram(shaderProgram);
}

void specifySceneVertexAttributes(GLuint shaderProgram)
{
    GLint posAttrib = glGetAttribLocation(shaderProgram, "position");
    glEnableVertexAttribArray(posAttrib);
    glVertexAttribPointer(posAttrib, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(GLfloat), 0);

    GLint colAttrib = glGetAttribLocation(shaderProgram, "color");
    glEnableVertexAttribArray(colAttrib);
    glVertexAttribPointer(colAttrib, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(GLfloat), (void*)(3 * sizeof(GLfloat)));

    GLint texAttrib = glGetAttribLocation(shaderProgram, "texcoord");
    glEnableVertexAttribArray(texAttrib);
    glVertexAttribPointer(texAttrib, 2, GL_FLOAT, GL_FALSE, 8 * sizeof(GLfloat), (void*)(6 * sizeof(GLfloat)));
}

void specifyScreenVertexAttributes(GLuint shaderProgram)
{
    GLint posAttrib = glGetAttribLocation(shaderProgram, "position");
    glEnableVertexAttribArray(posAttrib);
    glVertexAttribPointer(posAttrib, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(GLfloat), 0);

    GLint texAttrib = glGetAttribLocation(shaderProgram, "texcoord");
    glEnableVertexAttribArray(texAttrib);
    glVertexAttribPointer(texAttrib, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(GLfloat), (void*)(2 * sizeof(GLfloat)));
}

// Create a texture from an image file
GLuint loadTexture(const GLchar* path)
{
    GLuint texture;
    glGenTextures(1, &texture);

    int width, height;
    unsigned char* image;

    glBindTexture(GL_TEXTURE_2D, texture);
    image = SOIL_load_image(path, &width, &height, 0, SOIL_LOAD_RGB);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, image);
    SOIL_free_image_data(image);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    return texture;
}




opengl_render::opengl_render(glm::uvec2 const & size)
{
    // Create VAOs
    glGenVertexArrays(1, &vaoCube);
    glGenVertexArrays(1, &vaoQuad);

    // Load vertex data
    glGenBuffers(1, &vboCube);
    glGenBuffers(1, &vboQuad);

    glBindBuffer(GL_ARRAY_BUFFER, vboCube);
    glBufferData(GL_ARRAY_BUFFER, sizeof(cubeVertices), cubeVertices, GL_STATIC_DRAW);

    glBindBuffer(GL_ARRAY_BUFFER, vboQuad);
    glBufferData(GL_ARRAY_BUFFER, sizeof(quadVertices), quadVertices, GL_STATIC_DRAW);

    // Create shader programs
    createShaderProgram(sceneVertexSource, sceneFragmentSource, sceneVertexShader, sceneFragmentShader, sceneShaderProgram);
    createShaderProgram(screenVertexSource, screenFragmentSource, screenVertexShader, screenFragmentShader, screenShaderProgram);

    // Specify the layout of the vertex data
    glBindVertexArray(vaoCube);
    glBindBuffer(GL_ARRAY_BUFFER, vboCube);
    specifySceneVertexAttributes(sceneShaderProgram);

    glBindVertexArray(vaoQuad);
    glBindBuffer(GL_ARRAY_BUFFER, vboQuad);
    specifyScreenVertexAttributes(screenShaderProgram);

    // Load textures
    texKitten = loadTexture("sample.png");
    texPuppy = loadTexture("sample2.png");

    glUseProgram(sceneShaderProgram);
    glUniform1i(glGetUniformLocation(sceneShaderProgram, "texKitten"), 0);
    glUniform1i(glGetUniformLocation(sceneShaderProgram, "texPuppy"), 1);

    glUseProgram(screenShaderProgram);
    glUniform1i(glGetUniformLocation(screenShaderProgram, "texFramebuffer"), 0);

    uniModel = glGetUniformLocation(sceneShaderProgram, "model");

    // Create framebuffer
    glGenFramebuffers(1, &frameBuffer);
    glBindFramebuffer(GL_FRAMEBUFFER, frameBuffer);

    // Create texture to hold color buffer
    glGenTextures(1, &texColorBuffer);
    glBindTexture(GL_TEXTURE_2D, texColorBuffer);

    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, size.x, size.y, 0, GL_RGB, GL_UNSIGNED_BYTE, NULL);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, texColorBuffer, 0);

    // Create Renderbuffer Object to hold depth and stencil buffers
    glGenRenderbuffers(1, &rboDepthStencil);
    glBindRenderbuffer(GL_RENDERBUFFER, rboDepthStencil);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH24_STENCIL8, size.x, size.y);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_RENDERBUFFER, rboDepthStencil);

    // Set up projection
    glm::mat4 view = glm::lookAt(
        glm::vec3(2.5f, 2.5f, 2.0f),
        glm::vec3(0.0f, 0.0f, 0.0f),
        glm::vec3(0.0f, 0.0f, 1.0f)
    );

    glUseProgram(sceneShaderProgram);

    uniView = glGetUniformLocation(sceneShaderProgram, "view");
    glUniformMatrix4fv(uniView, 1, GL_FALSE, glm::value_ptr(view));

    glm::mat4 proj = glm::perspective(glm::radians(45.0f), float(size.x) / size.y, 1.0f, 10.0f);
    uniProj = glGetUniformLocation(sceneShaderProgram, "proj");
    glUniformMatrix4fv(uniProj, 1, GL_FALSE, glm::value_ptr(proj));

    uniColor = glGetUniformLocation(sceneShaderProgram, "overrideColor");

    t_start = std::chrono::high_resolution_clock::now();
}

void opengl_render::render()
{

    // Bind our framebuffer and draw 3D scene (spinning cube)
    glBindFramebuffer(GL_FRAMEBUFFER, frameBuffer);
    glBindVertexArray(vaoCube);
    glEnable(GL_DEPTH_TEST);
    glUseProgram(sceneShaderProgram);

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, texKitten);
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, texPuppy);

    // Clear the screen to white
    glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // Calculate transformation
    auto t_now = std::chrono::high_resolution_clock::now();
    float time = std::chrono::duration_cast<std::chrono::duration<float>>(t_now - t_start).count();

    glm::mat4 model;
    model = glm::rotate(
        model,
        time * glm::radians(180.0f),
        glm::vec3(0.0f, 0.0f, 1.0f)
    );
    glUniformMatrix4fv(uniModel, 1, GL_FALSE, glm::value_ptr(model));

    // Draw cube
    glDrawArrays(GL_TRIANGLES, 0, 36);

    glEnable(GL_STENCIL_TEST);

    // Draw floor
    glStencilFunc(GL_ALWAYS, 1, 0xFF);
    glStencilOp(GL_KEEP, GL_KEEP, GL_REPLACE);
    glStencilMask(0xFF);
    glDepthMask(GL_FALSE);
    glClear(GL_STENCIL_BUFFER_BIT);

    glDrawArrays(GL_TRIANGLES, 36, 6);

    // Draw cube reflection
    glStencilFunc(GL_EQUAL, 1, 0xFF);
    glStencilMask(0x00);
    glDepthMask(GL_TRUE);

    model = glm::scale(glm::translate(model, glm::vec3(0, 0, -1)), glm::vec3(1, 1, -1));
    glUniformMatrix4fv(uniModel, 1, GL_FALSE, glm::value_ptr(model));

    glUniform3f(uniColor, 0.3f, 0.3f, 0.3f);
    glDrawArrays(GL_TRIANGLES, 0, 36);
    glUniform3f(uniColor, 1.0f, 1.0f, 1.0f);

    glDisable(GL_STENCIL_TEST);

    // Bind default framebuffer and draw contents of our framebuffer
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glBindVertexArray(vaoQuad);
    glDisable(GL_DEPTH_TEST);
    glUseProgram(screenShaderProgram);

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, texColorBuffer);

    glDrawArrays(GL_TRIANGLES, 0, 6);
}

opengl_render::~opengl_render()
{
    glDeleteRenderbuffers(1, &rboDepthStencil);
    glDeleteTextures(1, &texColorBuffer);
    glDeleteFramebuffers(1, &frameBuffer);

    glDeleteTextures(1, &texKitten);
    glDeleteTextures(1, &texPuppy);

    glDeleteProgram(screenShaderProgram);
    glDeleteShader(screenFragmentShader);
    glDeleteShader(screenVertexShader);

    glDeleteProgram(sceneShaderProgram);
    glDeleteShader(sceneFragmentShader);
    glDeleteShader(sceneVertexShader);

    glDeleteBuffers(1, &vboCube);
    glDeleteBuffers(1, &vboQuad);

    glDeleteVertexArrays(1, &vaoCube);
    glDeleteVertexArrays(1, &vaoQuad);
}


void opengl_render::draw(size_t frame_count, std::function<void(GLuint)> on_render)
{
    for (size_t i = 0; i < frame_count; ++i)
    {
        render();
        on_render(texColorBuffer);
    }
}
