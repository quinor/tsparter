#include "tsparter/image_filters.hh"
#include "tsparter/timeit.hh"
#include "misc/avx_mathfun.h"

#include <cmath>
#include <x86intrin.h>

namespace ta
{

namespace
{


using Tensor3i = Eigen::Tensor<uint8_t, 3>;
using Tensor3f = Eigen::Tensor<float, 3>;


inline Tensor3f img_to_grayscale(const Tensor3i& e) noexcept
{
    ta::Timeit time("img_to_grayscale");
    return e
        .template cast<float>()
        .mean(Eigen::array<Eigen::Index, 1>{0})
        .reshape(Eigen::array<Eigen::Index, 3>{1, e.dimension(1), e.dimension(2)})
        .unaryExpr([](float val) {
            return val/255.f;
            // return log(1.f+val)/log(256.f);
        });
}

inline Tensor3i to_img(const Tensor3f& e) noexcept
{
    ta::Timeit time("to_img");
    return e
        .unaryExpr([](float val) {
            float tmp = val < 0.f ? 0.f : (val > 1.f ? 1.f : val);
            return tmp*255.f;
            // return exp(tmp*log(256.f))-1;
        })
        .template cast<uint8_t>()
        .broadcast(Eigen::array<Eigen::Index, 3>{3, 1, 1});
}

template <size_t M>
inline Tensor3f remap(const Tensor3f& e, const Tensor3f& m, float sigma, float alpha, float beta) noexcept
{
    ta::Timeit time("remap");

    static_assert(M%8 == 0, "M has to be divisible by 8!");
    size_t M8 = M/8;
    size_t N = e.size();

    __m256 ms[M8];
    Tensor3f ret(M, e.dimension(1), e.dimension(2));

    for (size_t i=0; i<M; i+=8)
        ms[i/8] = _mm256_load_ps(m.data() + i);

    const float* src = e.data();
    float* dst = ret.data();

    const __m256 mask0 = _mm256_set1_ps(-0.f); // sign mask
    const __m256 m_sigma = _mm256_set1_ps(sigma);
    const __m256 m_alpha = _mm256_set1_ps(alpha);
    const __m256 m_beta = _mm256_set1_ps(beta);
    const __m256 m_1 = _mm256_set1_ps(logf(sigma) * (1.f-alpha));
    const __m256 m_2 = _mm256_set1_ps(sigma*(1.f-beta));
    #pragma omp parallel for
    for (size_t i=0; i<N; i++)
    {
        for (size_t j=0; j<M8; j++)
        {
            __m256 m = ms[j];
            __m256 x = _mm256_set1_ps(src[i]) - m;
            __m256 xs = _mm256_and_ps(mask0, x);
            __m256 xa = _mm256_andnot_ps(mask0, x);

            __m256 ret = m + _mm256_or_ps(_mm256_mask_blend_ps(
                _mm256_cmp_ps_mask(xa, m_sigma, _CMP_LT_OS),
                _mm256_fmadd_ps(xa, m_beta, m_2),
                exp256_ps(log256_ps(xa+1e-6f)*m_alpha + m_1)
            ), xs);
            _mm256_store_ps(&dst[j*8 + i*M], ret);
        }
    }

    return ret;
}

template <size_t M>
inline Tensor3f downscale(const Tensor3f& input) noexcept
{
    ta::Timeit time("downscale<M>");

    static_assert(M%8 == 0, "M has to be divisible by 8!");
    size_t W = input.dimension(1), H = input.dimension(2);
    size_t W2 = (W+1)/2, H2 = (H+1)/2;

    Tensor3f tmp((int)M, (int)W, (int)H2);

    const float* src = input.data();
    float* dst = tmp.data();
    #pragma omp parallel for
    for (size_t j=0; j<W; j++)
        for (size_t k=0; k<M; k+=8)
        {
            const __m256 f3 = _mm256_set1_ps(3.f);

            __m256 x1 = _mm256_load_ps(src + k + j*M + 0*M*W);
            __m256 x2 = x1;
            for (size_t i=0; i<H2-1; i++)
            {
                __m256 x3 = _mm256_load_ps(src + k + j*M + (i*2+1)*M*W);
                __m256 x4 = _mm256_load_ps(src + k + j*M + (i*2+2)*M*W);
                _mm256_store_ps(dst + k + j*M + i*M*W, x1*0.125f + x2*0.375f + x3*0.375f + x4*0.125f);
                x1 = x3;
                x2 = x4;
            }
            __m256 x3 = _mm256_load_ps(src + k + j*M + std::min(H2*2-1, H-1)*M*W);
            __m256 x4 = x3;
            _mm256_store_ps(dst + k + j*M + (H2-1)*M*W, x1*0.125f + x2*0.375f + x3*0.375f + x4*0.125f);
        }

    Tensor3f ret((int)M, (int)W2, (int)H2);

    src = tmp.data();
    dst = ret.data();
    #pragma omp parallel for
    for (size_t i=0; i<H2; i++)
        for (size_t k=0; k<M; k+=8)
        {
            __m256 x1 = _mm256_load_ps(src + k + 0*M + i*M*W);
            __m256 x2 = x1;
            for (size_t j=0; j<W2-1; j++)
            {
                __m256 x3 = _mm256_load_ps(src + k + (j*2+1)*M + i*M*W);
                __m256 x4 = _mm256_load_ps(src + k + (j*2+2)*M + i*M*W);
                _mm256_store_ps(dst + k + j*M + i*M*W2, x1*0.125f + x2*0.375f + x3*0.375f + x4*0.125f);
                x1 = x3;
                x2 = x4;
            }
            __m256 x3 = _mm256_load_ps(src + k + std::min(W2*2-1, W-1)*M + i*M*W);
            __m256 x4 = x3;
            _mm256_store_ps(dst + k + (W2-1)*M + i*M*W2, x1*0.125f + x2*0.375f + x3*0.375f + x4*0.125f);
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
void mm256_store_op(__m256 val, float* addr)
{
    static_assert(OP < 3, "0-3 operations supported only");
    if constexpr (OP == 0)
        _mm256_store_ps(addr, val);
    else
    {
        __m256 old = _mm256_load_ps(addr);
        if constexpr (OP == 1)
            _mm256_store_ps(addr, old+val);
        else
            _mm256_store_ps(addr, old-val);
    }
}

template<size_t OP>
void store_op(float val, float* addr)
{
    static_assert(OP < 3, "0-3 operations supported only");
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

        // I ignore the borders. Fuck the borders. TODO don't fuck.
        const float* src = input.data();
        float* dst = output.data();
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
    }
    else
    {
        ta::Timeit time("upscale<M>");

        static_assert(M%8 == 0, "M has to be divisible by 8!");
        size_t W = output.dimension(1), H = output.dimension(2);
        size_t W2 = input.dimension(1), H2 = input.dimension(2);

        const float* src = input.data();
        float* dst = output.data();

        // I ignore the borders. Fuck the borders. TODO don't fuck.
        #pragma omp parallel for
        for (size_t i=1; i<2*H2-1; i+=2)
            for (size_t j=1; j<2*W2-1; j+=2)
                for (size_t k=0; k<M; k+=8)
                {
                    __m256 lu = _mm256_load_ps(src + k + (j/2)*M + (i/2)*M*W2);
                    __m256 ru = _mm256_load_ps(src + k + (j/2+1)*M + (i/2)*M*W2);
                    __m256 ld = _mm256_load_ps(src + k + (j/2)*M + (i/2+1)*M*W2);
                    __m256 rd = _mm256_load_ps(src + k + (j/2+1)*M + (i/2+1)*M*W2);

                    const float d9 = 9.f/16.f, d3 = 3.f/16.f, d1 = 1.f/16.f;
                    mm256_store_op<OP>(lu*d9 + ru*d3 + ld*d3 + rd*d1, dst + k + j*M + i*M*W);
                    mm256_store_op<OP>(lu*d3 + ru*d9 + ld*d1 + rd*d3, dst + k + (j+1)*M + i*M*W);
                    mm256_store_op<OP>(lu*d3 + ru*d1 + ld*d9 + rd*d3, dst + k + j*M + (i+1)*M*W);
                    mm256_store_op<OP>(lu*d1 + ru*d3 + ld*d3 + rd*d9, dst + k + (j+1)*M + (i+1)*M*W);
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

Eigen::Tensor<uint8_t, 3> pyramid(
    const Eigen::Tensor<uint8_t, 3>& input, float sigma, float alpha, float beta)
{
    ta::Timeit time("pyramid");
    const size_t N = 12;
    const size_t M = 64; // M % 8 == 0
    static_assert(M%8 == 0, "M has to be divisible by 8!");

    std::vector<Tensor3f> stack_base, stack_transformed;
    stack_base.reserve(N+1);
    stack_transformed.reserve(N+1);

    Tensor3f mids(M, 1, 1);
    for (size_t i=0; i<M; i++)
        mids(i, 0, 0) = i/float(M-1);

    time.checkpoint("nothing");

    stack_base.push_back(img_to_grayscale(input));
    stack_transformed.push_back(remap<M>(
        stack_base[0],
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
    return to_img(ret.slice(
        Eigen::array<Eigen::Index, 3>{0, 0, 0},
        Eigen::array<Eigen::Index, 3>{1, ret.dimension(1), ret.dimension(2)}
    ));
}

}
