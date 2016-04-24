cmake_minimum_required(VERSION 3.2)

project(rendeing-server)

add_subdirectory(encoder)
add_subdirectory(render)

include_directories(
	${CMAKE_SOURCE_DIR}/include
	${GL_INCLUDES}
)

link_directories(
	${GL_LINK_DIRS}
)

add_executable(${PROJECT_NAME}
	main.cpp
)

target_link_libraries(${PROJECT_NAME}
	${OPENGL_LIBRARIES}
	encoder
	render
	glew
	glfw3
)