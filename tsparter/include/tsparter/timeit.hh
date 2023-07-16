#pragma once
#include <chrono>
#include <iostream>
#include <vector>
#include <string>

namespace ta
{

class Timeit
{
    using time_point = std::chrono::time_point<std::chrono::system_clock>;
public:
    Timeit(const char* label)
    : _start(std::chrono::system_clock::now())
    , _label(label)
    {}

    void checkpoint(const char* label)
    {
        _intermediate.emplace_back(std::chrono::system_clock::now(), label);
    }

    ~Timeit()
    {
        using namespace std::literals;
        auto end = std::chrono::system_clock::now();
        _intermediate.emplace_back(end, "<tail>");
        auto elapsed = (end - _start)/1ms;

        // don't clog with short ones
        if (elapsed < 5 || !PRINT_TIMINGS)
            return;

        std::cout << _label << ": elapsed " << elapsed << " miliseconds" << std::endl;
        auto prev = _start;

        // if there's no checkpoints, don't print them
        if (_intermediate.size() == 1)
            return;
        for (auto& e : _intermediate)
        {
            std::cout << "  " << _label << "." << e.second
                << ": elapsed " << (e.first-prev)/1ms << " miliseconds" << std::endl;
            prev = e.first;
        }
    }

private:
    const static bool PRINT_TIMINGS = true;
    const time_point _start;
    std::string _label;
    std::vector<std::pair<time_point, std::string>> _intermediate;
};

}

