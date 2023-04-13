find_package(OpenMP REQUIRED)

add_library(tsparter
    stb_impl.cc
    image_util.cc
    image_filters.cc

    include/tsparter/image_util.hh
    include/tsparter/image_filters.hh
    include/tsparter/timeit.hh
    misc/avx_mathfun.h
)
target_include_directories(tsparter PRIVATE .)
target_include_directories(tsparter PUBLIC include)

target_link_libraries(tsparter PUBLIC stb Eigen3::Eigen OpenMP::OpenMP_CXX)
target_compile_options(tsparter PUBLIC -march=skylake-avx512)
