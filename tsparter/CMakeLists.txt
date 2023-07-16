find_package(OpenMP REQUIRED)

find_package(Sanitizers REQUIRED)

add_library(tsparter
    stb_impl.cc
    image_util.cc
    image_pyramid.cc
    image_tone_mapping.cc

    include/tsparter/image_util.hh
    include/tsparter/image_pyramid.hh
    include/tsparter/image_tone_mapping.hh
    include/tsparter/image_bits.hh
    include/tsparter/timeit.hh
    include/external/tinyfiledialogs.h

    external/sse_mathfun.h
    external/tinyfiledialogs.c
)
target_include_directories(tsparter PRIVATE .)
target_include_directories(tsparter PUBLIC include)

target_link_libraries(tsparter PUBLIC stb Eigen3::Eigen OpenMP::OpenMP_CXX)

target_compile_options(tsparter PUBLIC -march=haswell)

# add_sanitizers(tsparter)
