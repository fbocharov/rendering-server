#Top-level cmake-list

cmake_minimum_required(VERSION 3.2)

project(rendering-server)

find_package(CUDA   REQUIRED)
find_package(OpenGL REQUIRED)

set(GL_INCLUDES
	${CMAKE_SOURCE_DIR}/thirdparty/opengl/include
	${CMAKE_SOURCE_DIR}/thirdparty/glm/include
)

set(GL_LINK_DIRS
	${CMAKE_SOURCE_DIR}/thirdparty/opengl/lib/x64
)
add_definitions(/DGLEW_STATIC)


add_subdirectory(src)
