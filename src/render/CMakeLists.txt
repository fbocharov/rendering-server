cmake_minimum_required(VERSION 3.2)

project(render)

include_directories(
	${CMAKE_SOURCE_DIR}/include
	${GL_INCLUDES}
)

file(GLOB_RECURSE SOURCES "*.cpp")

link_directories(
	${GL_LINK_DIRS}
)

add_library(${PROJECT_NAME}
	${SOURCES}
)

target_link_libraries(${PROJECT_NAME}
	${OPENGL_LIBRARIES}
	glew
	glfw3
)