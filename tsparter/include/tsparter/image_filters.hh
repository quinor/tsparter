#pragma once
#include "tsparter/image_bits.hh"

namespace ta
{

Tensor3f pyramid(
    const Tensor3f& input, float sigma, float alpha, float beta);

}

