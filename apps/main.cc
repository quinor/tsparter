#include "gui.hh"
#include <tsparter/image_util.hh>
#include <tsparter/image_filters.hh>
#include <external/tinyfiledialogs.h>
#include <future>


using namespace std::chrono_literals;


void show_image(unsigned int tex, ImVec2 space, ImVec2 shape, bool center=false)
{
    ImVec2 ratio = space/shape;
    float scale = std::min(ratio.x, ratio.y);
    ImVec2 zero_pos = ImGui::GetCursorPos();

    if (center)
        ImGui::SetCursorPos(zero_pos + (space-shape*scale)/2);
    ImGui::Image((ImTextureID)(uint64_t)tex, shape*scale);
    if (center)
        ImGui::SetCursorPos(zero_pos + ImVec2(0, space.y));
}

bool show_image_button(const char* id, unsigned int tex, ImVec2 space, ImVec2 shape, bool center=false)
{
    ImVec2 ratio = space/shape;
    float scale = std::min(ratio.x, ratio.y);
    ImVec2 zero_pos = ImGui::GetCursorPos();

    if (center)
        ImGui::SetCursorPos(zero_pos + (space-shape*scale)/2);
    bool ret = ImGui::ImageButton(id, (ImTextureID)(uint64_t)tex, shape*scale);
    if (center)
        ImGui::SetCursorPos(zero_pos + ImVec2(0, space.y));
    return ret;
}

int main (int argc, char** argv)
{
    unsigned int textures[3];

    Tensor3f image_raw;
    Tensor3f image_filtered;
    // Tensor3f image_output;
    ImVec2 image_shape = {1, 1};

    auto recompute_raw = [&](const char* filename)
    {
        image_raw = ta::img_to_grayscale(ta::load_image(filename));
        image_shape = {(float)image_raw.dimension(1), (float)image_raw.dimension(2)};
        load_texture_from_tensor(textures[0], ta::to_img(image_raw));
    };

    auto init = [&]()
    {
        generate_textures(3, textures);
        auto empty = ta::load_image("misc/empty.png");
        load_texture_from_tensor(textures[0], empty);
        load_texture_from_tensor(textures[1], empty);
        load_texture_from_tensor(textures[2], empty);
    };

    auto draw = [&]()
    {
        static int big_display_idx = 0;
        int temp_big_display_idx = -1;

        const size_t slider_flags = ImGuiSliderFlags_AlwaysClamp;

        ImGuiIO& io = ImGui::GetIO();

        ImGui::SetNextWindowPos(ImVec2(0.0f, 0.0f));
        ImGui::SetNextWindowSize(ImGui::GetIO().DisplaySize);
        ImGui::Begin("main", nullptr,
            ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove);

        ImGui::BeginGroup();

        ImVec2 child_window_size = {
            240,
            (ImGui::GetContentRegionAvail().y - 2*ImGui::GetStyle().ItemSpacing.y)/3.f
        };

        static bool image_load_dirty = true;
        static bool pyramid_dirty = true;

        if(ImGui::BeginChild("loading", child_window_size, true, ImGuiWindowFlags_None))
        {
            bool show_file_dialog = false;

            if(show_image_button("loading", textures[0], {ImGui::GetContentRegionAvail().x, 120}, image_shape))
                big_display_idx = 0;

            if (ImGui::IsMouseDoubleClicked(0))
                show_file_dialog = true;
            if (ImGui::IsItemHovered())
                temp_big_display_idx = 0;

            static std::string filepath = argc == 1 ? "res/klaudia.jpg" : argv[1];
            size_t from;
            for (
                from = filepath.size()-1;
                from>0 && filepath[from] != '/' && filepath[from] != '\\'; from--
            );
            if (from == 0 || from == filepath.size() - 1)
                from = 0;
            else
                from++;

            std::string filename = filepath.substr(from, filepath.size()-from);

            static bool image_load_dirty = true;

            if (ImGui::Button(filename.c_str(), {ImGui::GetContentRegionAvail().x, 0}))
                show_file_dialog = true;

            if (show_file_dialog)
            {
                static std::vector<const char*> filePatterns = {
                    "*.png", "*.jpg", "*.jpeg", "*.bmp",
                    "*.PNG", "*.JPG", "*.JPEG", "*.BMP"
                };

                auto result = tinyfd_openFileDialog(
                        "Choose an image",
                        ".",
                        filePatterns.size(),
                        filePatterns.data(),
                        "Images",
                        0
                );

                if (result != nullptr)
                {
                    filepath = result;
                    image_load_dirty = true;
                }
            }

            if (image_load_dirty)
            {
                // Needs not to be async because it's fast? IDK if it will hold
                recompute_raw(filepath.c_str());
                image_load_dirty = false;
                pyramid_dirty = true;
            }
        }
        ImGui::EndChild();

        if(ImGui::BeginChild("preprocessing", child_window_size, true, ImGuiWindowFlags_None))
        {
            if(show_image_button("preprocessing", textures[1], {ImGui::GetContentRegionAvail().x, 120}, image_shape))
                big_display_idx = 1;
            if (ImGui::IsItemHovered())
                temp_big_display_idx = 1;

            const float
                alpha_slide_default = 0,
                beta_slide_default = 0,
                sigma_slide_default = 1;
            static float
                alpha_slide = 1,
                beta_slide = -1,
                sigma_slide = 2;

            bool changed = false;

            changed |= ImGui::SliderFloat("##alpha", &alpha_slide, -1.f, 1.0f, "%.2f", slider_flags);
            ImGui::SameLine();
            if (ImGui::Button("texture", {ImGui::GetContentRegionAvail().x, 0}))
                alpha_slide = alpha_slide_default, changed = true;
            if (ImGui::IsItemHovered(ImGuiHoveredFlags_DelayNormal)) ImGui::SetTooltip(
                "Click to reset the slider.\n\n"
                "Manipulate the strength of fine details in the image."
            );

            changed |= ImGui::SliderFloat("##beta", &beta_slide, -1.f, 1.0f, "%.2f", slider_flags);
            ImGui::SameLine();
            if (ImGui::Button("clarity", {ImGui::GetContentRegionAvail().x, 0}))
                beta_slide = beta_slide_default, changed = true;
            if (ImGui::IsItemHovered(ImGuiHoveredFlags_DelayNormal)) ImGui::SetTooltip(
                "Click to reset the slider.\n\n"
                "Manipulate the dynamic range of big features in the image"
            );

            changed |= ImGui::SliderFloat("##sigma", &sigma_slide, 0, 2.f, "%.2f", slider_flags);
            ImGui::SameLine();
            if (ImGui::Button("threshold", {ImGui::GetContentRegionAvail().x, 0}))
                sigma_slide = sigma_slide_default, changed = true;
            if (ImGui::IsItemHovered(ImGuiHoveredFlags_DelayNormal)) ImGui::SetTooltip(
                "Click to reset the slider.\n\n"
                "Choose the threshold between what's considered a detail (affected by texture)\n"
                "and what's a big feature (affected by clarity)"
            );

            float sigma = sigma_slide*0.15;
            float alpha = expf(-alpha_slide*logf(4));
            float beta = beta_slide + 1;

            static std::future<Tensor3f> done;
            static bool waiting = false;

            if (changed)
            {
                big_display_idx = 1;
                pyramid_dirty = true;
            }

            if (!done.valid() || done.wait_for(0s) == std::future_status::ready)
            {
                if (waiting)
                {
                    image_filtered = done.get();
                    load_texture_from_tensor(textures[1], ta::to_img(image_filtered));
                    waiting = false;
                }
                if (pyramid_dirty)
                {
                    done = std::async(std::launch::async, ta::pyramid, image_raw, sigma, alpha, beta);
                    waiting = true;
                    pyramid_dirty = false;
                }

            }

        }
        ImGui::EndChild();

        if(ImGui::BeginChild("rendering", child_window_size, true, ImGuiWindowFlags_None))
        {
            if(show_image_button("rendering", textures[2], {ImGui::GetContentRegionAvail().x, 120}, image_shape))
                big_display_idx = 2;
             if (ImGui::IsItemHovered())
                temp_big_display_idx = 2;
       }
        ImGui::EndChild();

        ImGui::EndGroup();
        ImGui::SameLine();

        if(ImGui::BeginChild("preview", {0, 0}, true, ImGuiWindowFlags_None))
        {
            int idx = temp_big_display_idx >= 0 ? temp_big_display_idx : big_display_idx;
            show_image(textures[idx], ImGui::GetContentRegionAvail(), image_shape, true);
        }
        ImGui::EndChild();


        ImGui::End();
    };

    return window_loop("TSPArter", init, draw);
}
