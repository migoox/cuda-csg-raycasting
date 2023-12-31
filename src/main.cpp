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

#include "camera.h"

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
void update_camera(camera::FPCamera& camera, float dt) {
    static glm::vec2 last_mouse_pos = glm::vec2(Window::get_mouse_pos().first, Window::get_mouse_pos().second);
    float sensitivity = 0.02f;
    glm::vec2 mouse_pos = glm::vec2(Window::get_mouse_pos().first, Window::get_mouse_pos().second);
    glm::vec2 mouse_delta = (mouse_pos - last_mouse_pos) * sensitivity;
    last_mouse_pos = mouse_pos;

    if (!Window::is_mouse_btn_pressed(GLFW_MOUSE_BUTTON_RIGHT)) {
        glfwSetInputMode(Window::get_win_handle(), GLFW_CURSOR, GLFW_CURSOR_NORMAL);
        return;
    }

    glfwSetInputMode(Window::get_win_handle(), GLFW_CURSOR, GLFW_CURSOR_DISABLED);

    float speed = 0.3f;
    float r = 0.f, u = 0.f, f = 0.f;

    if (Window::is_key_pressed(GLFW_KEY_W)) {
        f += speed * dt;
    }
    else if (Window::is_key_pressed(GLFW_KEY_S)) {
        f -= speed * dt;
    }

    if (Window::is_key_pressed(GLFW_KEY_A)) {
        r -= speed * dt;
    }
    else if (Window::is_key_pressed(GLFW_KEY_D)) {
        r += speed * dt;
    }

    if (Window::is_key_pressed(GLFW_KEY_Q)) {
        u -= speed * dt;
    }
    else if (Window::is_key_pressed(GLFW_KEY_E)) {
        u += speed * dt;
    }

    camera.rotate(-mouse_delta.y * 0.03f, -mouse_delta.x * 0.03f);
    camera.move(r, u, f);
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
    camera::FPCamera camera(supp_res[selected_res].first, supp_res[selected_res].second, glm::radians(45.f), 0.1f, 500.f);

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
        update_camera(camera, 0.1f);

        glm::vec2 canvas_size(static_cast<float>(canvas.get_width()), static_cast<float>(canvas.get_height()));
        for (int y = 0; y < canvas.get_height(); y++) {
            for (int x = 0; x < canvas.get_width(); x++) {
                canvas.set_pixel(x, y, per_pixel(x, y, canvas_size, camera.get_pos(), camera.get_inverse_proj(), camera.get_inverse_view()));
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
