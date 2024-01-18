cmake_minimum_required(VERSION 3.26)

# Allow cuda and cpp languages
project(cuda-csg-raycasting LANGUAGES CXX CUDA)

# Set cpp standard
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)

# Find CUDA package
find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=sm_70")
# Find patterns with these patterns and add them to sources
file(GLOB SOURCES
        "src/*.cpp"
        "src/*.hpp"
        "src/*.h"
        "src/*.cu"
        "src/*.cuh"
        "src/vendor/stbi_image.h"
        "src/vendor/nlohmann/json.hpp"
        "src/vendor/nlohmann/json_fwd.hpp"
)

add_subdirectory(src/vendor)

# Set include and lib directories
include_directories(dependencies/include)

# Add lib directory
if(WIN32)
    set(LIB_DIR ${CMAKE_SOURCE_DIR}/dependencies/lib/win)
    file(GLOB EXTERNAL_LIBS "${LIB_DIR}/*.lib")
elseif(UNIX)
    set(LIB_DIR ${CMAKE_SOURCE_DIR}/dependencies/lib/linux)
    file(GLOB EXTERNAL_LIBS "${LIB_DIR}/*.so")
else()
    message(FATAL_ERROR "Unsupported platform")
endif()

# Create an executable from the source files
add_executable(cuda-csg-raycasting ${SOURCES})
# Link the executable to the external libraries
target_link_libraries(cuda-csg-raycasting
    ImGui
    ${EXTERNAL_LIBS}
)

# Generate separate object files for each CUDA source file
set_target_properties(cuda-csg-raycasting PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# Include CUDA directories
include_directories(${CUDA_INCLUDE_DIRS})

# WINDOWS POSTBUILD -- copy the .dll files into Relase/Debug folder
if (WIN32)
    if ($ENV{CLION_IDE})
         add_custom_command(
            TARGET ${PROJECT_NAME} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy
                ${CMAKE_CURRENT_BINARY_DIR}/../dependencies/bin/glfw3.dll
                ${CMAKE_CURRENT_BINARY_DIR}/../dependencies/bin/glew32.dll
                ${CMAKE_CURRENT_BINARY_DIR}
        )
    else()
        add_custom_command(
            TARGET ${PROJECT_NAME} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy
                ${CMAKE_CURRENT_BINARY_DIR}/dependencies/bin/glfw3.dll
                ${CMAKE_CURRENT_BINARY_DIR}/dependencies/bin/glew32.dll
                ${CMAKE_CURRENT_BINARY_DIR}/$<IF:$<CONFIG:Debug>,Debug,Release>
        )
        add_custom_command(
            TARGET ${PROJECT_NAME} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_directory
				${CMAKE_CURRENT_BINARY_DIR}/res
                ${CMAKE_CURRENT_BINARY_DIR}/$<IF:$<CONFIG:Debug>,Debug,Release>
        )
    endif()
endif()


