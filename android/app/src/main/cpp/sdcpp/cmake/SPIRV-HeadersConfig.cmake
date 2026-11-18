# Minimal CONFIG-mode stand-in for KhronosGroup/SPIRV-Headers. ggml-vulkan only calls
# find_package(SPIRV-Headers CONFIG REQUIRED) and #includes <spirv/unified1/spirv.hpp>; the headers come
# from the Android NDK's bundled copy (see SDCPP_SPIRV_HEADERS_INCLUDE in ../CMakeLists.txt).
if(NOT TARGET SPIRV-Headers::SPIRV-Headers)
  add_library(SPIRV-Headers::SPIRV-Headers INTERFACE IMPORTED)
  set_target_properties(SPIRV-Headers::SPIRV-Headers PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${SDCPP_SPIRV_HEADERS_INCLUDE}")
endif()
set(SPIRV-Headers_FOUND TRUE)
