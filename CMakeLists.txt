#Top-level cmake-list

cmake_minimum_required(VERSION 3.2)

project(rendering-server)

find_package(CUDA   REQUIRED)
find_package(OpenGL REQUIRED)

link_directories(${CUDA_TOOLKIT_ROOT_DIR}/lib/x64)

set(GL_INCLUDES
	${CMAKE_SOURCE_DIR}/thirdparty/opengl/include
	${CMAKE_SOURCE_DIR}/thirdparty/glm/include
)

if (CMAKE_BUILD_TYPE EQUAL "RELEASE")
	set(GL_LINK_DIRS
		${CMAKE_SOURCE_DIR}/thirdparty/opengl/lib/x64/Release
	)
else()
	set(GL_LINK_DIRS
		${CMAKE_SOURCE_DIR}/thirdparty/opengl/lib/x64/Debug
	)
endif()

message(STATUS "gl link dirs: " ${GL_LINK_DIRS})

add_definitions(/DGLEW_STATIC)


add_subdirectory(src)
