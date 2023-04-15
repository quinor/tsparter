#pragma once
#include "tsparter/image_bits.hh"

namespace ta
{

Tensor3f tone_mapping(
    const Tensor3f& input,
    float exposure, float contrast,
    float blacks, float shadows, float highlights, float whites
);

}
