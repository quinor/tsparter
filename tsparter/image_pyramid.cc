#include "tsparter/image_pyramid.hh"
#include "tsparter/timeit.hh"
#include "external/sse_mathfun.h"

#include <cmath>
#include <x86intrin.h>

namespace ta
{

namespace
{

template <size_t M>
inline Tensor3f remap(const Tensor3f& e, const Tensor3f& m, float sigma, float alpha, float beta) noexcept
{
    ta::Timeit time("remap");

    static_assert(M%4 == 0, "M has to be divisible by 4!");
    size_t N = e.size();

    const size_t LUT_SIZE = 256;

    float lut[LUT_SIZE];

    for (size_t i=0; i<LUT_SIZE; i++)
    {
        float x = i/float(LUT_SIZE-1);
        lut[i] = x<sigma
            ? expf(logf(x+1e-6f)*alpha + logf(sigma)*(1.f-alpha))
            : x*beta + sigma*(1.f-beta)
        ;
    }

    __m128 ms[M/4];
    Tensor3f ret(M, e.dimension(1), e.dimension(2));

    for (size_t i=0; i<M; i+=4)
        ms[i/4] = _mm_load_ps(m.data() + i);

    const float* src = e.data();
    float* dst = ret.data();

    const __m128 mask0 = _mm_set1_ps(-0.f); // sign mask
    #pragma omp parallel for
    for (size_t i=0; i<N; i++)
    {
        for (size_t j=0; j<M; j+=4)
        {
            __m128 m = ms[j/4];
            __m128 x = _mm_set1_ps(src[i]) - m;
            __m128 xs = _mm_and_ps(mask0, x);
            __m128 xa = _mm_andnot_ps(mask0, x);

            __m128 ret = m + _mm_or_ps(_mm_i32gather_ps(
                lut,
                _mm_cvtps_epi32(
                    _mm_round_ps(xa*float(LUT_SIZE-1),
                    _MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC
                )),
                sizeof(float)
            ), xs);
            _mm_store_ps(&dst[j + i*M], ret);
        }
    }

    return ret;
}

template <size_t M>
inline Tensor3f downscale(const Tensor3f& input) noexcept
{
    ta::Timeit time("downscale<M>");

    static_assert(M%4 == 0, "M has to be divisible by 4!");
    size_t W = input.dimension(1), H = input.dimension(2);
    size_t W2 = (W+1)/2, H2 = (H+1)/2;

    Tensor3f tmp((int)M, (int)W, (int)H2);

    const float* src = input.data();
    float* dst = tmp.data();
    #pragma omp parallel for
    for (size_t j=0; j<W; j++)
        for (size_t k=0; k<M; k+=4)
        {
            const __m128 f3 = _mm_set1_ps(3.f);

            __m128 x1 = _mm_load_ps(src + k + j*M + 0*M*W);
            __m128 x2 = x1;
            for (size_t i=0; i<H2-1; i++)
            {
                __m128 x3 = _mm_load_ps(src + k + j*M + (i*2+1)*M*W);
                __m128 x4 = _mm_load_ps(src + k + j*M + (i*2+2)*M*W);
                _mm_store_ps(dst + k + j*M + i*M*W, x1*0.125f + x2*0.375f + x3*0.375f + x4*0.125f);
                x1 = x3;
                x2 = x4;
            }
            __m128 x3 = _mm_load_ps(src + k + j*M + std::min(H2*2-1, H-1)*M*W);
            __m128 x4 = x3;
            _mm_store_ps(dst + k + j*M + (H2-1)*M*W, x1*0.125f + x2*0.375f + x3*0.375f + x4*0.125f);
        }

    Tensor3f ret((int)M, (int)W2, (int)H2);

    src = tmp.data();
    dst = ret.data();
    #pragma omp parallel for
    for (size_t i=0; i<H2; i++)
        for (size_t k=0; k<M; k+=4)
        {
            __m128 x1 = _mm_load_ps(src + k + 0*M + i*M*W);
            __m128 x2 = x1;
            for (size_t j=0; j<W2-1; j++)
            {
                __m128 x3 = _mm_load_ps(src + k + (j*2+1)*M + i*M*W);
                __m128 x4 = _mm_load_ps(src + k + (j*2+2)*M + i*M*W);
                _mm_store_ps(dst + k + j*M + i*M*W2, x1*0.125f + x2*0.375f + x3*0.375f + x4*0.125f);
                x1 = x3;
                x2 = x4;
            }
            __m128 x3 = _mm_load_ps(src + k + std::min(W2*2-1, W-1)*M + i*M*W);
            __m128 x4 = x3;
            _mm_store_ps(dst + k + (W2-1)*M + i*M*W2, x1*0.125f + x2*0.375f + x3*0.375f + x4*0.125f);
        }

    return ret;
}

template <>
inline Tensor3f downscale<1>(const Tensor3f& input) noexcept
{
    ta::Timeit time("downscale<1>");
    size_t W = input.dimension(1), H = input.dimension(2);
    size_t W2 = (W+1)/2, H2 = (H+1)/2;

    Tensor3f tmp(1, (int)W, (int)H2);

    const float* src = input.data();
    float* dst = tmp.data();
    #pragma omp parallel for
    for (size_t j=0; j<W; j++)
        {
            float x1 = src[j + 0*W];
            float x2 = x1;
            for (size_t i=0; i<H2-1; i++)
            {
                float x3 = src[j + (i*2+1)*W];
                float x4 = src[j + (i*2+2)*W];
                dst[j + i*W] = x1*0.125f + x2*0.375f + x3*0.375f + x4*0.125f;
                x1 = x3;
                x2 = x4;
            }
            float x3 = src[j + std::min(H2*2-1, H-1)*W];
            float x4 = x3;
            dst[j + (H2-1)*W] = x1*0.125f + x2*0.375f + x3*0.375f + x4*0.125f;
        }

    Tensor3f ret(1, (int)W2, (int)H2);

    src = tmp.data();
    dst = ret.data();
    #pragma omp parallel for
    for (size_t i=0; i<H2; i++)
    {
        float x1 = src[0+i*W];
        float x2 = x1;
        for (size_t j=0; j<W2-1; j++)
        {
            float x3 = src[j*2+1 + i*W];
            float x4 = src[j*2+2 + i*W];
            dst[j + i*W2] = x1*0.125f + x2*0.375f + x3*0.375f + x4*0.125f;
            x1 = x3;
            x2 = x4;
        }
        float x3 = src[std::min(W2*2-1, W-1) + i*W];
        float x4 = x3;
        dst[W2-1 + i*W2] = x1*0.125f + x2*0.375f + x3*0.375f + x4*0.125f;
    }

    return ret;
}

// OP is:
// - 0 - write
// - 1 - add
// - 2 - sub

template<size_t OP>
void mm128_store_op(__m128 val, float* addr)
{
    static_assert(OP < 3, "0-2 operations supported only");
    if constexpr (OP == 0)
        _mm_store_ps(addr, val);
    else
    {
        __m128 old = _mm_load_ps(addr);
        if constexpr (OP == 1)
            _mm_store_ps(addr, old+val);
        else
            _mm_store_ps(addr, old-val);
    }
}

template<size_t OP>
void store_op(float val, float* addr)
{
    static_assert(OP < 3, "0-2 operations supported only");
    if constexpr (OP == 0)
        *addr = val;
    else
    {
        if constexpr (OP == 1)
            *addr += val;
        else
            *addr -= val;
    }
}

template <size_t M, size_t OP>
inline void upscale(const Tensor3f& input, Tensor3f& output) noexcept
{
    if constexpr (M == 1)
    {
        ta::Timeit time("upscale<1>");

        size_t W = output.dimension(1), H = output.dimension(2);
        size_t W2 = input.dimension(1), H2 = input.dimension(2);

        float* dst = output.data();
        const float* src = input.data();

        #pragma omp parallel for
        for (size_t i=1; i<2*H2-1; i+=2)
            for (size_t j=1; j<2*W2-1; j+=2)
            {
                float lu = src[j/2 + (i/2)*W2];
                float ru = src[j/2+1 + (i/2)*W2];
                float ld = src[j/2 + (i/2+1)*W2];
                float rd = src[j/2+1 + (i/2+1)*W2];

                const float d9 = 9.f/16.f, d3 = 3.f/16.f, d1 = 1.f/16.f;
                store_op<OP>(lu*d9 + ru*d3 + ld*d3 + rd*d1, dst + j + i*W);
                store_op<OP>(lu*d3 + ru*d9 + ld*d1 + rd*d3, dst + j+1 + i*W);
                store_op<OP>(lu*d3 + ru*d1 + ld*d9 + rd*d3, dst + j + (i+1)*W);
                store_op<OP>(lu*d1 + ru*d3 + ld*d3 + rd*d9, dst + j+1 + (i+1)*W);
            }

        // I don't ignore the borders. It's annoying.
        store_op<OP>(src[0 + 0*W2], dst + 0 + 0*W);
        if (W%2 == 0) store_op<OP>(src[W2-1 + 0*W2], dst + W-1 + 0*W);
        if (H%2 == 0) store_op<OP>(src[0 + (H2-1)*W2], dst + 0 + (H-1)*W);
        if (W%2 == 0 && H%2 == 0) store_op<OP>(src[W2-1 + (H2-1)*W2], dst + W-1 + (H-1)*W);

        for (size_t j=1; j<2*W2-1; j+=2)
        {
            float l, r;
            const float d3 = 3.f/4.f, d1 = 1.f/4.f;

            l = src[j/2 + 0*W2];
            r = src[j/2+1 + 0*W2];
            store_op<OP>(l*d3 + r*d1, dst + j + 0*W);
            store_op<OP>(l*d1 + r*d3, dst + j+1 + 0*W);

            if (H%2 == 0)
            {
                l = src[j/2 + (H2-1)*W2];
                r = src[j/2+1 + (H2-1)*W2];
                store_op<OP>(l*d3 + r*d1, dst + j + (H-1)*W);
                store_op<OP>(l*d1 + r*d3, dst + j+1 + (H-1)*W);
            }
        }
        for (size_t i=1; i<2*H2-1; i+=2)
        {
            float u, d;
            const float d3 = 3.f/4.f, d1 = 1.f/4.f;

            u = src[0 + (i/2)*W2];
            d = src[0 + (i/2+1)*W2];
            store_op<OP>(u*d3 + d*d1, dst + 0 + i*W);
            store_op<OP>(u*d1 + d*d3, dst + 0 + (i+1)*W);

            if (W%2 == 0)
            {
                u = src[W2-1 + (i/2)*W2];
                d = src[W2-1 + (i/2+1)*W2];
                store_op<OP>(u*d3 + d*d1, dst + W-1 + i*W);
                store_op<OP>(u*d1 + d*d3, dst + W-1 + (i+1)*W);
            }
        }
    }
    else
    {
        ta::Timeit time("upscale<M>");

        static_assert(M%4 == 0, "M has to be divisible by 4!");
        size_t W = output.dimension(1), H = output.dimension(2);
        size_t W2 = input.dimension(1), H2 = input.dimension(2);

        float* dst = output.data();
        const float* src = input.data();

        #pragma omp parallel for
        for (size_t i=1; i<2*H2-1; i+=2)
            for (size_t j=1; j<2*W2-1; j+=2)
                for (size_t k=0; k<M; k+=4)
                {
                    __m128 lu = _mm_load_ps(src + k + (j/2)*M + (i/2)*M*W2);
                    __m128 ru = _mm_load_ps(src + k + (j/2+1)*M + (i/2)*M*W2);
                    __m128 ld = _mm_load_ps(src + k + (j/2)*M + (i/2+1)*M*W2);
                    __m128 rd = _mm_load_ps(src + k + (j/2+1)*M + (i/2+1)*M*W2);

                    const float d9 = 9.f/16.f, d3 = 3.f/16.f, d1 = 1.f/16.f;
                    mm128_store_op<OP>(lu*d9 + ru*d3 + ld*d3 + rd*d1, dst + k + j*M + i*M*W);
                    mm128_store_op<OP>(lu*d3 + ru*d9 + ld*d1 + rd*d3, dst + k + (j+1)*M + i*M*W);
                    mm128_store_op<OP>(lu*d3 + ru*d1 + ld*d9 + rd*d3, dst + k + j*M + (i+1)*M*W);
                    mm128_store_op<OP>(lu*d1 + ru*d3 + ld*d3 + rd*d9, dst + k + (j+1)*M + (i+1)*M*W);
                }

        // I don't ignore the borders. It's annoying.
        for (size_t k=0; k<M; k+=4)
        {
            mm128_store_op<OP>(
                _mm_load_ps(src + k + M*(0 + 0*W2)), dst + k + M*(0 + 0*W));
            if (W%2 == 0) mm128_store_op<OP>(
                _mm_load_ps(src + k + M*(W2-1 + 0*W2)), dst + k + M*(W-1 + 0*W));
            if (H%2 == 0) mm128_store_op<OP>(
                _mm_load_ps(src + k + M*(0 + (H2-1)*W2)), dst + k + M*(0 + (H-1)*W));
            if (W%2 == 0 && H%2 == 0) mm128_store_op<OP>(
                _mm_load_ps(src + k + M*(W2-1 + (H2-1)*W2)), dst + k + M*(W-1 + (H-1)*W));

            for (size_t j=1; j<2*W2-1; j+=2)
            {
                __m128 l, r;
                const float d3 = 3.f/4.f, d1 = 1.f/4.f;

                l = _mm_load_ps(src + k + M*(j/2 + 0*W2));
                r = _mm_load_ps(src + k + M*(j/2+1 + 0*W2));
                mm128_store_op<OP>(l*d3 + r*d1, dst + k + M*(j + 0*W));
                mm128_store_op<OP>(l*d1 + r*d3, dst + k + M*(j+1 + 0*W));

                if (H%2 == 0)
                {
                    l = _mm_load_ps(src + k + M*(j/2 + (H2-1)*W2));
                    r = _mm_load_ps(src + k + M*(j/2+1 + (H2-1)*W2));
                    mm128_store_op<OP>(l*d3 + r*d1, dst + k + M*(j + (H-1)*W));
                    mm128_store_op<OP>(l*d1 + r*d3, dst + k + M*(j+1 + (H-1)*W));
                }
            }
            for (size_t i=1; i<2*H2-1; i+=2)
            {
                __m128 u, d;
                const float d3 = 3.f/4.f, d1 = 1.f/4.f;

                u = _mm_load_ps(src + k + M*(0 + (i/2)*W2));
                d = _mm_load_ps(src + k + M*(0 + (i/2+1)*W2));
                mm128_store_op<OP>(u*d3 + d*d1, dst + k + M*(0 + i*W));
                mm128_store_op<OP>(u*d1 + d*d3, dst + k + M*(0 + (i+1)*W));

                if (W%2 == 0)
                {
                    u = _mm_load_ps(src + k + M*(W2-1 + (i/2)*W2));
                    d = _mm_load_ps(src + k + M*(W2-1 + (i/2+1)*W2));
                    mm128_store_op<OP>(u*d3 + d*d1, dst + k + M*(W-1 + i*W));
                    mm128_store_op<OP>(u*d1 + d*d3, dst + k + M*(W-1 + (i+1)*W));
                }
            }
        }
    }
}

}

template <size_t M>
inline void combine(const Tensor3f& transformed, Tensor3f& base)
{
    ta::Timeit time("combine");

    const size_t N = transformed.dimension(1)*transformed.dimension(2);
    const float* src = transformed.data();
    float* dst = base.data();

    #pragma omp parallel for
    for (size_t i=0; i<N; i++)
    {
        float x = dst[i]*(float)(M-1);
        int low = floor(x), high = ceil(x);
        float frac = x-low;
        float a = src[i*M + low], b = src[i*M + high];
        dst[i] = a + (b-a)*frac;
    }
}

template <size_t M>
Tensor3f pyramid(const Tensor3f& input, float sigma, float alpha, float beta)
{
    static_assert(M%4 == 0, "M has to be divisible by 4!");

    ta::Timeit time("pyramid");
    const size_t N = 12;

    std::vector<Tensor3f> stack_base, stack_transformed;
    stack_base.reserve(N+1);
    stack_transformed.reserve(N+1);

    Tensor3f mids(M, 1, 1);
    for (size_t i=0; i<M; i++)
        mids(i, 0, 0) = i/float(M-1);

    time.checkpoint("nothing");

    stack_base.push_back(input);
    stack_transformed.push_back(remap<M>(
        input,
        mids,
        sigma, alpha, beta
    ));

    time.checkpoint("transform");

    for (size_t i=0; i<N; i++)
    {
        stack_transformed.push_back(downscale<M>(stack_transformed[i]));
        upscale<M, 2>(stack_transformed[i+1], stack_transformed[i]);
        stack_base.push_back(downscale<1>(stack_base[i]));
        combine<M>(stack_transformed[i], stack_base[i]);
    }

    time.checkpoint("erect");

    for (size_t i=N; i>0; i--)
        upscale<1, 1>(stack_base[i], stack_base[i-1]);

    time.checkpoint("collapse");

    const Tensor3f& ret = stack_base[0];
    return ret.slice(
        Eigen::array<Eigen::Index, 3>{0, 0, 0},
        Eigen::array<Eigen::Index, 3>{1, ret.dimension(1), ret.dimension(2)}
    );
}

template
Tensor3f pyramid<16>(const Tensor3f& input, float sigma, float alpha, float beta);

template
Tensor3f pyramid<64>(const Tensor3f& input, float sigma, float alpha, float beta);

}
