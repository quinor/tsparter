#pragma once

#include <functional>
#define IMGUI_DEFINE_MATH_OPERATORS
#include <imgui.h>
#include <unsupported/Eigen/CXX11/Tensor>

void generate_textures(int n, unsigned int* textures);
void load_texture_from_tensor(unsigned int texture, const Eigen::Tensor<uint8_t, 3>& data);
// init is called once before drawing but after initializing opengl
// draw is called every event loop
int window_loop(const char* window_name, std::function<void()> init, std::function<void()> draw);
