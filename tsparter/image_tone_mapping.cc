#include "tsparter/image_tone_mapping.hh"
#include "tsparter/timeit.hh"

#include <cmath>
#include <x86intrin.h>

namespace ta
{

namespace
{

inline float smoothstep(float x)
{
    return x * x * x * (x * (x * 6 - 15) + 10);
}

inline float bell(float x, float low, float high)
{
    x = x < low ? low : (x > high ? high : x);
    x = 2.f*(x-low)/(high-low);
    return x < 1.f ? smoothstep(x) : smoothstep(2-x);
}

}

Tensor3f tone_mapping(
    const Tensor3f& input,
    float exposure, float contrast,
    float blacks, float shadows, float highlights, float whites
)
{
    ta::Timeit time("tone_mapping");

    size_t RES=1023;
    float lut[RES+1];

    for (size_t i=0; i<=RES; i++)
    {
        float val = i/float(RES);
        val = powf(val, exposure);
        val = std::copysignf(powf(2.f*fabsf(val-0.5f), contrast)/2.f, val-0.5f) + 0.5f;
        val = val
            + blacks * bell(val, 0, 0.5)
            + shadows * bell(val, 0, 0.75)
            + highlights * bell(val, 0.25, 1)
            + whites * bell(val, 0.5, 1)
        ;
        lut[i] = val < 0.f ? 0.f : (val > 1.f ? 1.f : val);
    }

    // explicitly materialize to avoid UB by having the lambda access lut after the outer
    // function had exitted
    Tensor3f ret = input.unaryExpr([&](float val) {
        int x = RES*val;
        x = x < 0 ? 0 : (x > RES ? RES : x);
        return lut[x];
    });
    return ret;
}

}
