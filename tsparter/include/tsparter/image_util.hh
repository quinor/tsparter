#pragma once
#include <unsupported/Eigen/CXX11/Tensor>

namespace ta
{

// loads in hwc format
Eigen::Tensor<uint8_t, 3> load_image(const std::string& filename);

}

