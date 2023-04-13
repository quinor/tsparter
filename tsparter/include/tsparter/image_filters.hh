#pragma once
#include <unsupported/Eigen/CXX11/Tensor>

namespace ta
{

Eigen::Tensor<uint8_t, 3> pyramid(
    const Eigen::Tensor<uint8_t, 3>& input, float sigma, float alpha, float beta);

}

