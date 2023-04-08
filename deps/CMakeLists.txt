# Automatically update submodules as needed
find_package(Git QUIET)
if(GIT_FOUND AND EXISTS "${PROJECT_SOURCE_DIR}/.git")
    option(GIT_SUBMODULE "Check submodules during build" ON)
    if(GIT_SUBMODULE)
        message(STATUS "Submodule update")
        execute_process(COMMAND ${GIT_EXECUTABLE} submodule update --init --recursive
                        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                        RESULT_VARIABLE GIT_SUBMOD_RESULT)
        if(NOT GIT_SUBMOD_RESULT EQUAL "0")
            message(FATAL_ERROR "git submodule update --init --recursive failed with ${GIT_SUBMOD_RESULT}, please checkout submodules")
        endif()
    endif()
endif()


# test for all dependencies
if(
    NOT EXISTS "${PROJECT_SOURCE_DIR}/deps/xtl/CMakeLists.txt" OR
    NOT EXISTS "${PROJECT_SOURCE_DIR}/deps/xtensor/CMakeLists.txt" OR
    NOT EXISTS "${PROJECT_SOURCE_DIR}/deps/xsimd/CMakeLists.txt" OR
    NOT EXISTS "${PROJECT_SOURCE_DIR}/deps/imgui/imgui.h" OR
    NOT EXISTS "${PROJECT_SOURCE_DIR}/deps/stb/stb_image.h" # arbitrary file
)
    message(FATAL_ERROR "The submodules were not downloaded! GIT_SUBMODULE was turned off or failed. Please update submodules and try again.")
endif()

# add dependencies which are already cmake
add_subdirectory(xtl)
add_subdirectory(xsimd)
add_subdirectory(xtensor)

# add simple integration for header-only stb
add_library(stb INTERFACE)
target_include_directories(stb INTERFACE ${CMAKE_CURRENT_LIST_DIR}/stb)


# add more involved integration for imgui
set (imgui_SOURCE_DIR ${CMAKE_CURRENT_LIST_DIR}/imgui)
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
