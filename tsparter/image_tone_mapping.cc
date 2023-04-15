#include "tsparter/image_tone_mapping.hh"
#include "tsparter/timeit.hh"

#include <cmath>
#include <x86intrin.h>

namespace ta
{

Tensor3f tone_mapping(
    const Tensor3f& input,
    float exposure, float contrast,
    float blacks, float shadows, float highlights, float whites
)
{
    ta::Timeit time("tone_mapping");
    return input.unaryExpr([&](float val) {
        val = powf(val, exposure);
        val = std::copysignf(powf(2.f*fabsf(val-0.5f), contrast)/2.f, val-0.5f) + 0.5f;
        return val;
    });
}

}



/*
s : 0...1
s+contrast*s*(1-s)

s == t+1

t+1 + contrast*(t+1)*t

x == (1/(t+1))
x : 0...1

x + contrast*x*(1-x)
1/(t+1) + contrast * 1/(t+1) * (t/(t+1))
*/
