#include "tsparter/image_util.hh"
#include "tsparter/timeit.hh"
#include <stb_image.h>
#include <cstdio>
#include <cstdlib>


namespace ta
{

using Tensor3i = Eigen::Tensor<uint8_t, 3>;
using Tensor3f = Eigen::Tensor<float, 3>;

Tensor3i load_image(const std::string& filename)
{
    ta::Timeit time("load_image");
    int width, height, nrChannels;
    unsigned char *data = stbi_load(filename.c_str(), &width, &height, &nrChannels, 3);
    if (data == nullptr)
    {
        printf("ERROR: failed to load image: %s\n", filename.c_str());
        std::exit(-1);
    }

    Eigen::Tensor<uint8_t, 3> ret = Eigen::TensorMap<Eigen::Tensor<uint8_t, 3>>(data, 3, width, height);
    stbi_image_free(data);
    return ret;
}

Tensor3f img_to_grayscale(const Tensor3i& e)
{
    ta::Timeit time("img_to_grayscale");
    return e
        .template cast<float>()
        .mean(Eigen::array<Eigen::Index, 1>{0})
        .reshape(Eigen::array<Eigen::Index, 3>{1, e.dimension(1), e.dimension(2)})
        .unaryExpr([](float val) {
            return val/255.f;
        });
}

Tensor3i to_img(const Tensor3f& e)
{
    ta::Timeit time("to_img");
    return e
        .unaryExpr([](float val) {
            float tmp = val < 0.f ? 0.f : (val > 1.f ? 1.f : val);
            return tmp*255.f;
        })
        .template cast<uint8_t>()
        .broadcast(Eigen::array<Eigen::Index, 3>{3, 1, 1});
}

// TODO: save_image

}
