#pragma once

#include <functional>
#include <imgui.h>

int window_loop(const char* window_name, std::function<void(ImGuiIO&)> draw);
