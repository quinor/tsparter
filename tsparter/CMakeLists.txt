add_library(tsparter
    stb_impl.cc
    image_load.cc
    image_load.hh
)
target_link_libraries(tsparter xtensor xtensor::optimize xtensor::use_xsimd)
