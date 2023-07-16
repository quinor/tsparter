#include "gui.hh"
#include <tsparter/image_util.hh>
#include <tsparter/image_pyramid.hh>
#include <tsparter/image_tone_mapping.hh>
#include <external/tinyfiledialogs.h>
#include <future>


using namespace std::chrono_literals;


static float em = 0;

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

bool show_image_button(const char* id, unsigned int tex, ImVec2 space, ImVec2 shape)
{
    ImVec2 ratio = space/shape;
    float scale = std::min(ratio.x, ratio.y);
    bool ret = ImGui::ImageButton(id, (ImTextureID)(uint64_t)tex, shape*scale);
    return ret;
}

bool show_slider(const char* tag, float* var, float min, float max, float default_val, const char* tooltip)
{
    ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x - 6*em);
    bool ret = ImGui::SliderFloat(tag, var, min, max, "%.2f", ImGuiSliderFlags_AlwaysClamp);
    if (ImGui::IsItemHovered() && ImGui::IsMouseDoubleClicked(0))
    {
        *var = default_val;
        ret = true;
    }
    if (ImGui::IsItemHovered(ImGuiHoveredFlags_DelayNormal))
        ImGui::SetTooltip(tooltip);
    return ret;
}


int main (int argc, char** argv)
{
    unsigned int textures[3];

    Tensor3f image_raw;
    Tensor3f image_filtered;
    Tensor3f image_mapped;
    // Tensor3f image_output;
    ImVec2 image_shape = {1, 1};

    auto recompute_raw = [&](const char* filename)
    {
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
        const size_t slider_flags = ImGuiSliderFlags_AlwaysClamp;

        ImGuiIO& io = ImGui::GetIO();

        em = ImGui::GetFontSize();

        ImGui::SetNextWindowPos(ImVec2(0.0f, 0.0f));
        ImGui::SetNextWindowSize(io.DisplaySize);
        ImGui::Begin("main", nullptr,
            ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove);

        ImGui::BeginGroup();

        static int big_display_idx = 0;
        int temp_big_display_idx = -1;

        static bool image_load_dirty = true;
        static bool pyramid_lo_dirty = false;
        static bool pyramid_hi_dirty = false;
        static bool mapping_dirty = false;

        if(ImGui::BeginChild("loading", {18*em, 9*em}, true, ImGuiWindowFlags_None))
        {
            bool show_file_dialog = false;

            if(show_image_button("loading", textures[0], {ImGui::GetContentRegionAvail().x, 6*em}, image_shape))
                big_display_idx = 0;

            if (ImGui::IsItemHovered() && ImGui::IsMouseDoubleClicked(0))
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

            // Needs not to be async because it's fast? IDK if it will hold
            if (image_load_dirty)
            {
                image_raw = ta::img_to_grayscale(ta::load_image(filepath.c_str()));
                image_shape = {(float)image_raw.dimension(1), (float)image_raw.dimension(2)};
                load_texture_from_tensor(textures[0], ta::to_img(image_raw));
                image_load_dirty = false;
                pyramid_lo_dirty = true;
            }
        }
        ImGui::EndChild();

        if(ImGui::BeginChild("preprocessing", {18*em, 21.5f*em}, true, ImGuiWindowFlags_None))
        {
            if(show_image_button("preprocessing", textures[1], {ImGui::GetContentRegionAvail().x, 6*em}, image_shape))
                big_display_idx = 1;
            if (ImGui::IsItemHovered())
                temp_big_display_idx = 1;

            const float
                alpha_slide_default = 0,
                beta_slide_default = 0,
                sigma_slide_default = 1,
                exposure_slide_default = 0,
                contrast_slide_default = 0,
                blacks_slide_default = 0,
                shadows_slide_default = 0,
                highlights_slide_default = 0,
                whites_slide_default = 0;

            static float
                alpha_slide = 0.5,
                beta_slide = -0.5,
                sigma_slide = 1,
                exposure_slide = 0,
                contrast_slide = 0.5,
                blacks_slide = 0,
                shadows_slide = 0,
                highlights_slide = 0,
                whites_slide = 0;

            bool pyramid_changed = false;
            bool mapping_changed = false;

            pyramid_changed |= show_slider(
                "texture", &alpha_slide,
                -1.f, 1.0f, alpha_slide_default,
                "Double click on the label to reset the slider.\n\n"
                "Manipulate the strength of fine details in the image."
            );

            pyramid_changed |= show_slider(
                "clarity", &beta_slide,
                -1.f, 1.0f, beta_slide_default,
                "Double click on the label to reset the slider.\n\n"
                "Manipulate the dynamic range of big features in the image"
            );

            pyramid_changed |= show_slider(
                "threshold", &sigma_slide,
                0.f, 2.0f, sigma_slide_default,
                "Double click on the label to reset the slider.\n\n"
                "Choose the threshold between what's considered a detail (affected by texture)\n"
                "and what's a big feature (affected by clarity)"
            );

            mapping_changed |= show_slider(
                "exposure", &exposure_slide,
                -1.f, 1.0f, exposure_slide_default,
                "Double click on the label to reset the slider."
            );

            mapping_changed |= show_slider(
                "contrast", &contrast_slide,
                -1.f, 1.0f, contrast_slide_default,
                "Double click on the label to reset the slider."
            );

            mapping_changed |= show_slider(
                "blacks", &blacks_slide,
                -1.f, 1.0f, blacks_slide_default,
                "Double click on the label to reset the slider."
            );

            mapping_changed |= show_slider(
                "shadows", &shadows_slide,
                -1.f, 1.0f, shadows_slide_default,
                "Double click on the label to reset the slider."
            );

            mapping_changed |= show_slider(
                "highlights", &highlights_slide,
                -1.f, 1.0f, highlights_slide_default,
                "Double click on the label to reset the slider."
            );

            mapping_changed |= show_slider(
                "whites", &whites_slide,
                -1.f, 1.0f, whites_slide_default,
                "Double click on the label to reset the slider."
            );


            if (pyramid_changed)
            {
                big_display_idx = 1;
                pyramid_lo_dirty = true;
            }

            if (mapping_changed)
            {
                big_display_idx = 1;
                mapping_dirty = true;
            }

            static std::future<Tensor3f> pyramid_lo_done;
            if (!pyramid_lo_done.valid() || pyramid_lo_done.wait_for(0s) == std::future_status::ready)
            {
                static bool waiting = false;
                if (waiting)
                {
                    image_filtered = pyramid_lo_done.get();
                    waiting = false;
                    // dirty the next steps here
                    mapping_dirty = true;
                    pyramid_hi_dirty = true;
                }
                if (pyramid_lo_dirty && image_raw.size())
                {
                    float
                        sigma = sigma_slide*0.15,
                        alpha = expf(-alpha_slide*logf(4)),
                        beta = beta_slide + 1;

                    // copy the input so that recomputing the previous step will not produce artifacts
                    static Tensor3f input_image;
                    input_image = image_raw;

                    pyramid_lo_done = std::async(std::launch::async, ta::pyramid<16>, input_image, sigma, alpha, beta);
                    waiting = true;
                    pyramid_lo_dirty = false;
                }
            }

            static std::future<Tensor3f> pyramid_hi_done;
            if (!pyramid_hi_done.valid() || pyramid_hi_done.wait_for(0s) == std::future_status::ready)
            {
                static bool waiting = false;
                if (waiting)
                {
                    image_filtered = pyramid_hi_done.get();
                    waiting = false;
                    // dirty the next steps here
                    mapping_dirty = true;
                }
                if (pyramid_hi_dirty && image_raw.size())
                {
                    float
                        sigma = sigma_slide*0.15,
                        alpha = expf(-alpha_slide*logf(4)),
                        beta = beta_slide + 1;

                    // copy the input so that recomputing the previous step will not produce artifacts
                    static Tensor3f input_image;
                    input_image = image_raw;

                    pyramid_hi_done = std::async(std::launch::async, ta::pyramid<64>, input_image, sigma, alpha, beta);
                    waiting = true;
                    pyramid_hi_dirty = false;
                }
            }

            static std::future<Tensor3f> mapping_done;
            if (!mapping_done.valid() || mapping_done.wait_for(0s) == std::future_status::ready)
            {
                static bool waiting = false;
                if (waiting)
                {
                    image_mapped = mapping_done.get();
                    load_texture_from_tensor(textures[1], ta::to_img(image_mapped));
                    waiting = false;
                    // dirty the next step here
                }
                if (mapping_dirty && image_filtered.size())
                {
                    float
                        exposure = powf(2.f, -2*exposure_slide),
                        contrast = powf(2.f, -2*contrast_slide),
                        blacks = 0.125*blacks_slide,
                        shadows = 0.25*shadows_slide,
                        highlights = 0.25*highlights_slide,
                        whites = 0.125*whites_slide;

                    // copy the input so that recomputing the previous step will not produce artifacts
                    static Tensor3f input_image;
                    input_image = image_filtered;

                    mapping_done = std::async(
                        std::launch::async, ta::tone_mapping, input_image,
                        exposure, contrast,
                        blacks, shadows, highlights, whites
                    );
                    waiting = true;
                    mapping_dirty = false;
                }
            }
        }
        ImGui::EndChild();

        if(ImGui::BeginChild("rendering", {18*em, ImGui::GetContentRegionAvail().y}, true, ImGuiWindowFlags_None))
        {
            if(show_image_button("rendering", textures[2], {ImGui::GetContentRegionAvail().x, 6*em}, image_shape))
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
