#pragma once
#include "tsparter/image_bits.hh"

namespace ta
{

// use with M=16 and M=64 only
template<size_t M>
Tensor3f pyramid(const Tensor3f& input, float sigma, float alpha, float beta);

}

