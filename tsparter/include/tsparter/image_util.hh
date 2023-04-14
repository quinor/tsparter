#pragma once
#include "tsparter/image_bits.hh"

namespace ta
{

// loads in hwc format
Eigen::Tensor<uint8_t, 3> load_image(const std::string& filename);

Tensor3f img_to_grayscale(const Tensor3i& e);
Tensor3i to_img(const Tensor3f& e);

}

