#include "tsparter/image_util.hh"
#include <stb_image.h>
#include <cstdio>
#include <cstdlib>


namespace ta
{

Eigen::Tensor<uint8_t, 3> load_image(const std::string& filename)
{
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

// TODO: save_image

}
