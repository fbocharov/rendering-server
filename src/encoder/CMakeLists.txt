cmake_minimum_required(VERSION 3.2)

project(encoder)

include_directories(
	${CMAKE_SOURCE_DIR}/include
	${CUDA_INCLUDE_DIRS}
)

file(GLOB_RECURSE CPP_SOURCES "*.cpp")
file(GLOB_RECURSE CUDA_SOURCES "*.cu")

set(CUDA_ATTACH_VS_BUILD_RULE_TO_CUDA_FILE ON)
set(CUDA_SEPARABLE_COMPILATION ON)

cuda_add_library(${PROJECT_NAME}
	${CPP_SOURCES}
	${CUDA_SOURCES}
)

target_link_libraries(${PROJECT_NAME}
	${CUDA_LIBRARIES}
	cuda
)