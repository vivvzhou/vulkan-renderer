cmake_minimum_required(VERSION 3.24)
project(vulkan-renderer LANGUAGES CXX)

# ---------------------------------------------------------------------------
# Language / output
# ---------------------------------------------------------------------------
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Debug CACHE STRING "" FORCE)
endif()

# ---------------------------------------------------------------------------
# Vulkan SDK (headers, loader, glslc)
# ---------------------------------------------------------------------------
find_package(Vulkan REQUIRED)
message(STATUS "Vulkan include : ${Vulkan_INCLUDE_DIRS}")
message(STATUS "Vulkan library : ${Vulkan_LIBRARIES}")
message(STATUS "glslc          : ${Vulkan_GLSLC_EXECUTABLE}")

# ---------------------------------------------------------------------------
# Third-party dependencies (fetched at configure time)
#   glfw, glm           -> built via their own CMake
#   vma, stb, tinygltf  -> header-only; populated and exposed as INTERFACE libs
#                          (SOURCE_SUBDIR points at a non-existent dir so their
#                           own CMake is NOT run)
# ---------------------------------------------------------------------------
include(FetchContent)

FetchContent_Declare(glfw
  GIT_REPOSITORY https://github.com/glfw/glfw.git
  GIT_TAG        3.4
  GIT_SHALLOW    TRUE)
set(GLFW_BUILD_DOCS     OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS    OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(GLFW_INSTALL        OFF CACHE BOOL "" FORCE)

FetchContent_Declare(glm
  GIT_REPOSITORY https://github.com/g-truc/glm.git
  GIT_TAG        1.0.1
  GIT_SHALLOW    TRUE)

FetchContent_Declare(stb
  GIT_REPOSITORY https://github.com/nothings/stb.git
  GIT_TAG        master)

FetchContent_Declare(tinygltf
  GIT_REPOSITORY https://github.com/syoyo/tinygltf.git
  GIT_TAG        v2.9.3
  GIT_SHALLOW    TRUE
  SOURCE_SUBDIR  __skip_cmake__)

FetchContent_MakeAvailable(glfw glm stb tinygltf)

# Sample model (single binary asset, downloaded but not extracted). Not committed to the repo.
FetchContent_Declare(model
  URL https://raw.githubusercontent.com/KhronosGroup/glTF-Sample-Assets/main/Models/ToyCar/glTF-Binary/ToyCar.glb
  DOWNLOAD_NO_EXTRACT TRUE)
FetchContent_MakeAvailable(model)
set(ASSET_PATH "${model_SOURCE_DIR}/ToyCar.glb")
message(STATUS "Model asset   : ${ASSET_PATH}")

# HDR environment map for image-based lighting (CC0, Poly Haven). Not committed.
set(ENV_HDR_NAME "autoshop_01_2k.hdr")
FetchContent_Declare(envhdr
  URL https://dl.polyhaven.org/file/ph-assets/HDRIs/hdr/2k/${ENV_HDR_NAME}
  DOWNLOAD_NO_EXTRACT TRUE)
FetchContent_MakeAvailable(envhdr)
set(ENV_HDR_PATH "${envhdr_SOURCE_DIR}/${ENV_HDR_NAME}")
message(STATUS "Env HDR asset : ${ENV_HDR_PATH}")

add_library(stb INTERFACE)
target_include_directories(stb INTERFACE ${stb_SOURCE_DIR})

add_library(tinygltf INTERFACE)
target_include_directories(tinygltf INTERFACE ${tinygltf_SOURCE_DIR})

# ---------------------------------------------------------------------------
# Shader compilation helper (glslc -> SPIR-V)
# ---------------------------------------------------------------------------
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/CompileShaders.cmake)

# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------
add_executable(vkrenderer
  src/main.cpp
  src/core/Window.cpp
  src/core/ThreadPool.cpp
  src/vk/Instance.cpp
  src/vk/Device.cpp
  src/vk/Swapchain.cpp
  src/vk/DeviceAllocator.cpp
  src/vk/Buffer.cpp
  src/vk/Image.cpp
  src/render/GltfLoader.cpp
  src/render/Ibl.cpp
  src/render/Renderer.cpp)

target_include_directories(vkrenderer PRIVATE src)

target_link_libraries(vkrenderer PRIVATE
  Vulkan::Vulkan
  glfw
  glm::glm
  stb
  tinygltf)

target_compile_definitions(vkrenderer PRIVATE
  GLM_FORCE_RADIANS
  GLM_FORCE_DEPTH_ZERO_TO_ONE
  GLFW_INCLUDE_VULKAN
  SHADER_DIR="${CMAKE_BINARY_DIR}/shaders"
  ASSET_PATH="${ASSET_PATH}"
  ENV_HDR_PATH="${ENV_HDR_PATH}"
  PIPELINE_CACHE_PATH="${CMAKE_BINARY_DIR}/pipeline_cache.bin"
  SSAO_WEIGHTS_PATH="${CMAKE_SOURCE_DIR}/assets/ssao_mlp.bin")

if(WIN32)
  target_compile_definitions(vkrenderer PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN)
endif()

if(MSVC)
  target_compile_options(vkrenderer PRIVATE /W4 /permissive-)
else()
  target_compile_options(vkrenderer PRIVATE -Wall -Wextra)
endif()

target_compile_shaders(vkrenderer
  SHADERS
    shaders/gbuffer.vert shaders/gbuffer.frag
    shaders/lighting.vert shaders/lighting.frag
    shaders/shadow.vert
    shaders/ibl_irradiance.comp
    shaders/ibl_prefilter.comp
    shaders/ibl_brdf.comp
    shaders/ssao.comp)
