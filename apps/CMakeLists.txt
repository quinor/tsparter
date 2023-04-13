add_library(gui STATIC
    gui.cc
    gui.hh
)
target_link_libraries(gui imgui imgui_backend Eigen3::Eigen tsparter)


# spawn applications
set(apps
    main.cc
)

set(libraries gui tsparter)

# Create each app into a binary with the same name.
# Have the binaries be outputted to the main directory of the project
# for ease of use.
foreach(app_src ${apps})
    get_filename_component(binary ${app_src} NAME_WE)
    add_executable(${binary} ${app_src})
    set_target_properties(${binary} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}
    )
    target_link_libraries(${binary} ${libraries})
endforeach()
