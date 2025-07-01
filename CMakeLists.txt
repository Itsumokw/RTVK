# CMakeList.txt : CMake project for rtvk, include source and define
# project specific logic here.
#

cmake_minimum_required(VERSION 3.12)
project(rtvk)

# Set GLFW paths
set(GLFW_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/glfw-3.4.bin.WIN64")
set(GLFW_INCLUDE_DIR "${GLFW_ROOT}/include")
set(GLFW_LIBRARY "${GLFW_ROOT}/lib-vc2022/glfw3.lib")
# set(TINY_OBJ_LOADER_INCLUDE_DIR "D:/vulkan/library/obj/")

# Add GLFW as imported library
add_library(glfw STATIC IMPORTED)
set_target_properties(glfw PROPERTIES
    IMPORTED_LOCATION "${GLFW_LIBRARY}"
    INTERFACE_INCLUDE_DIRECTORIES "${GLFW_INCLUDE_DIR}"
)

# Find Vulkan
find_package(Vulkan REQUIRED)

# 添加 ImGui 子目录
set(IMGUI_DIR "${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/imgui")  # 假设 ImGui 放在这里

# 添加 ImGui 核心源文件
file(GLOB IMGUI_SOURCES
    "${IMGUI_DIR}/*.cpp"
    "${IMGUI_DIR}/backends/imgui_impl_glfw.cpp"
    "${IMGUI_DIR}/backends/imgui_impl_vulkan.cpp"
)

# Add executable
add_executable(rtvk "samplevk.cpp" "samplevk.h" "tiny_obj_loader.h" ${IMGUI_SOURCES})

# Include Vulkan and GLFW directories
target_include_directories(rtvk PRIVATE
    ${Vulkan_INCLUDE_DIRS}
    ${GLFW_INCLUDE_DIR}
    ${IMGUI_DIR}
    ${IMGUI_DIR}/backends  # 关键：包含后端接口
)

# Link libraries
target_link_libraries(rtvk 
    ${Vulkan_LIBRARIES}
    glfw
)

# Set C++ standard
set_property(TARGET rtvk PROPERTY CXX_STANDARD 20)