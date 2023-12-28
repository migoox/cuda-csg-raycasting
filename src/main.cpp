#define GLM_FORCE_CUDA
#include <vector>
#include <glm/glm.hpp>

#include "imgui_internal.h"
#include "imgui.h"
#include "vendor/imgui/backend/imgui_impl_glfw.h"
#include "vendor/imgui/backend/imgui_impl_opengl3.h"
#include "gl_debug.h"
#include "image.hpp"
#include "window.hpp"

#include "cuda_ray_tracer.cuh"

#define GL_SILENCE_DEPRECATION
#if defined(IMGUI_IMPL_OPENGL_ES2)
#include <GLES2/gl2.h>
#endif
#include <GLFW/glfw3.h> // Will drag system OpenGL headers

#if defined(_MSC_VER) && (_MSC_VER >= 1900) && !defined(IMGUI_DISABLE_WIN32_FUNCTIONS)
#pragma comment(lib, "legacy_stdio_definitions")
#endif

std::vector<std::pair<int, int>> get_supported_resolutions() {
    std::vector<std::pair<int, int>> resolutions;
    resolutions.emplace_back(1920, 1080);
    resolutions.emplace_back(1440, 900);
    resolutions.emplace_back(1280, 720);
    resolutions.emplace_back(1024, 768);
    resolutions.emplace_back(640, 360);

    return resolutions;
}


// Main code
int main(int, char**)
{
    Window::init(1280, 720);

    ImGuiWindowFlags windowFlags = ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoResize;

    // Get supported resolutions
    std::vector<std::pair<int, int>> supp_res = get_supported_resolutions();
    char res_items[1024];
    {
        size_t curr_offset = 0;
        for (int i = 0; i < supp_res.size(); i++) {
            std::string curr_res = std::to_string(supp_res[i].first) + "x" + std::to_string(supp_res[i].second);
            strcpy(res_items + curr_offset, curr_res.c_str());
            curr_offset += curr_res.size() + 1;
        }
        res_items[curr_offset] = '\0';
    }
    int selected_res = supp_res.size() - 1;

    // INIT
    ImVec4 clear_color = ImVec4(0.45f, 0.55f, 0.60f, 1.00f);
    img::Image canvas(supp_res[selected_res].first, supp_res[selected_res].second);
    canvas.clear(0x0000FFFF);

    GLuint canvas_texture;
    GLCall(glGenTextures(1, &canvas_texture));
    GLCall(glBindTexture(GL_TEXTURE_2D, canvas_texture));
    GLCall(glPixelStorei(GL_UNPACK_ALIGNMENT, 1));
    GLCall(glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA,
                        canvas.get_width(), canvas.get_height(), 0,
                        GL_RGBA, GL_UNSIGNED_INT_8_8_8_8,
                        canvas.raw()));

    GLCall(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR));
    GLCall(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR));
    GLCall(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT));
    GLCall(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT));

    while (!Window::should_close())
    {
        glfwPollEvents();

        for (int y = 0; y < canvas.get_height(); y++) {
            for (int x = 0; x < canvas.get_width(); x++) {
                canvas.set_pixel(x, y, per_pixel(x, y));
            }
        }
        GLCall(glTexSubImage2D(GL_TEXTURE_2D, 0, 0,0,
                               canvas.get_width(), canvas.get_height(),
                        GL_RGBA, GL_UNSIGNED_INT_8_8_8_8,
                        canvas.raw()));

        // Start the Dear ImGui frame
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        auto height = static_cast<float>(Window::get_height());
        auto width_win1 = static_cast<float>(Window::get_width()) / 4.f * 3.f;
        auto width_win2 = static_cast<float>(Window::get_width()) / 4.f;
        ImVec2 win1_size = ImVec2(width_win1, height);
        ImVec2 win2_size = ImVec2(width_win2, height);

        if (Window::is_fullscreen()) {
            ImGui::SetNextWindowPos(ImVec2(0, 0));
            ImGui::SetNextWindowSize(ImVec2(Window::get_width(), Window::get_height()));
            win1_size = ImVec2(Window::get_width(), Window::get_height());
        } else {
            ImGui::SetNextWindowPos(ImVec2(0, 0));
            ImGui::SetNextWindowSize(win1_size);
        }

        ImGui::Begin("Window 1", nullptr, windowFlags);

        if (ImGui::Button("Toggle Fullscreen")) {
            Window::toggle_fullscreen();
        }

        // Display supported resolutions in a combo box
        ImGui::SameLine();
        if (ImGui::Combo("##Resolution", &selected_res, res_items, std::max(6, (int)supp_res.size()))) {
            canvas.resize(supp_res[selected_res].first, supp_res[selected_res].second);
            canvas.clear(0x0000FFFF);
            GLCall(glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA,
                        canvas.get_width(), canvas.get_height(), 0,
                        GL_RGBA, GL_UNSIGNED_INT_8_8_8_8,
                        canvas.raw()));
        }

        // Calculate the position to center the image
        ImVec2 img_size(canvas.get_width(), canvas.get_height());
        ImVec2 img_pos = ImGui::GetCursorScreenPos();
        img_pos.x += (win1_size.x - img_size.x) * 0.5f;
        img_pos.y += (win1_size.y - img_size.y) * 0.5f;

        // Display the image at the calculated position
        ImGui::SetCursorScreenPos(img_pos);
        ImGui::Image((void*)(intptr_t)canvas_texture, img_size);
        ImGui::End();

        if (!Window::is_fullscreen()) {
            ImGui::SetNextWindowPos(ImVec2(width_win1, 0));
            ImGui::SetNextWindowSize(win2_size);
            ImGui::Begin("Window 2", nullptr, windowFlags);

            ImGui::End();
        }

        // Rendering
        ImGui::Render();
        auto display = Window::get_framebuffer_size();

        glViewport(0, 0, display.first, display.second);
        glClearColor(clear_color.x, clear_color.y, clear_color.z, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);

        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        Window::swap_buffers();
    }
    return 0;
}
