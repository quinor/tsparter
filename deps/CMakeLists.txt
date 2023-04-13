include(FetchContent)

FetchContent_Declare(
  Eigen
  GIT_REPOSITORY    https://gitlab.com/libeigen/eigen.git
  GIT_TAG           3.4.0
)
FetchContent_MakeAvailable(Eigen)

FetchContent_Declare(
  imgui
  GIT_REPOSITORY https://github.com/ocornut/imgui.git
  GIT_TAG        v1.89.4
)

FetchContent_Declare(
  stb
  GIT_REPOSITORY https://github.com/nothings/stb.git
  GIT_TAG        5736b15f7ea0ffb08dd38af21067c314d6a3aae9
)

FetchContent_MakeAvailable(Eigen imgui stb)

# add simple integration for header-only stb
add_library(stb INTERFACE)
target_include_directories(stb INTERFACE ${stb_SOURCE_DIR})


# add more involved integration for imgui
add_library(imgui STATIC
    ${imgui_SOURCE_DIR}/imgui.h
    ${imgui_SOURCE_DIR}/imgui_internal.h
    ${imgui_SOURCE_DIR}/imgui.cpp
    ${imgui_SOURCE_DIR}/imgui_demo.cpp
    ${imgui_SOURCE_DIR}/imgui_draw.cpp
    ${imgui_SOURCE_DIR}/imgui_tables.cpp
    ${imgui_SOURCE_DIR}/imgui_widgets.cpp
)
target_include_directories(imgui PUBLIC
    ${imgui_SOURCE_DIR}/
    ${imgui_SOURCE_DIR}/backends
)


# for imgui_backend we need opengl and glfw
find_package(glfw3 REQUIRED)
find_package(OpenGL REQUIRED)

# ...and then it can get integrated
add_library(imgui_backend STATIC
    ${imgui_SOURCE_DIR}/backends/imgui_impl_glfw.cpp
    ${imgui_SOURCE_DIR}/backends/imgui_impl_opengl3.cpp
)
target_link_libraries(imgui_backend imgui glfw OpenGL::GL)
