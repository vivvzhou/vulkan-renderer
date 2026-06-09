# CompileShaders.cmake
# ---------------------------------------------------------------------------
# Compiles GLSL shaders to SPIR-V with glslc (shipped in the Vulkan SDK) and
# wires the resulting .spv files into a target as a build dependency.
#
#   target_compile_shaders(vkrenderer
#     SHADERS shaders/triangle.vert shaders/triangle.frag)
#
# .spv files are written to ${CMAKE_BINARY_DIR}/shaders/<name>.spv
# ---------------------------------------------------------------------------

function(target_compile_shaders TARGET_NAME)
  cmake_parse_arguments(ARG "" "" "SHADERS" ${ARGN})

  if(NOT Vulkan_GLSLC_EXECUTABLE)
    message(FATAL_ERROR
      "glslc not found (Vulkan_GLSLC_EXECUTABLE is empty). Install the Vulkan SDK.")
  endif()

  set(_spv_out_dir "${CMAKE_BINARY_DIR}/shaders")
  file(MAKE_DIRECTORY "${_spv_out_dir}")

  set(_spv_files "")
  foreach(_shader ${ARG_SHADERS})
    get_filename_component(_name "${_shader}" NAME)
    set(_spv "${_spv_out_dir}/${_name}.spv")
    add_custom_command(
      OUTPUT  "${_spv}"
      COMMAND ${Vulkan_GLSLC_EXECUTABLE} -O --target-env=vulkan1.3
              "${CMAKE_CURRENT_SOURCE_DIR}/${_shader}" -o "${_spv}"
      DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/${_shader}"
      COMMENT "glslc ${_shader} -> shaders/${_name}.spv"
      VERBATIM)
    list(APPEND _spv_files "${_spv}")
  endforeach()

  add_custom_target(${TARGET_NAME}_shaders DEPENDS ${_spv_files})
  add_dependencies(${TARGET_NAME} ${TARGET_NAME}_shaders)
endfunction()
