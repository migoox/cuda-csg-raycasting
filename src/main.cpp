#include <glm/glm.hpp>

#include <GL/glew.h>
#include "gl_debug.h"
#include "camera.hpp"
#include "shader_program.hpp"
#include "window.hpp"
#include "textures.hpp"

#include "camera_operator.hpp"
#include "imgui.h"
#include "vendor/imgui/backend/imgui_impl_glfw.h"
#include "vendor/imgui/backend/imgui_impl_opengl3.h"
#include "billboard.hpp"

#include "cpu_raytracer.hpp"
#include "csg.hpp"

#if defined(_MSC_VER) && (_MSC_VER >= 1900) && !defined(IMGUI_DISABLE_WIN32_FUNCTIONS)
#pragma comment(lib, "legacy_stdio_definitions")
#endif

using namespace renderer;

const int INIT_SCREEN_SIZE_X = 800;
const int INIT_SCREEN_SIZE_Y = 600;

void update_canvas(Image& canvas, app::CameraOperator& cam_operator) {
    for (int y = 0; y < canvas.get_height(); y++) {
        for (int x = 0; x < canvas.get_width(); x++) {
            canvas.set_pixel(x, y, cpu_raytracer::per_pixel(x, y,
                                                            glm::vec2(canvas.get_width(), canvas.get_height()),
                                                            cam_operator.get_cam().get_pos(),
                                                            cam_operator.get_cam().get_inv_proj(),
                                                            cam_operator.get_cam().get_inv_view()));
        }
    }
}

// Main code
int main(int, char**) {
    // Init GLFW
    Backend::init_glfw();
    {
        // Create a window
        Window window(INIT_SCREEN_SIZE_X, INIT_SCREEN_SIZE_Y, "OpenGL 3D scenery");

        // Init GLEW and ImGui
        Backend::init_glew();
        Backend::init_imgui(window);

        // Application state
        ImVec4 clear_color = ImVec4(0.45f, 0.55f, 0.60f, 1.00f);

        std::string executable_dir = std::filesystem::path(__FILE__).parent_path().string();
        csg::CSGTree tree = csg::CSGTree(executable_dir + "/../res/csg_tree_example1.json");
        ShaderProgram sh(executable_dir + "/../res/billboard.vert", executable_dir + "/../res/billboard.frag");
        sh.bind();
        sh.set_uniform_1i("u_texture", 0);

        Image canvas = Image(INIT_SCREEN_SIZE_X, INIT_SCREEN_SIZE_Y, 0xFF0000FF);
        auto txt_res = std::make_shared<TextureResource>(canvas);
        Texture txt = Texture(txt_res);

        app::CameraOperator cam_operator((float)INIT_SCREEN_SIZE_X, (float)INIT_SCREEN_SIZE_Y, glm::radians(45.f), 0.1f, 1000.f);
        app::Billboard billboard;

        std::chrono::steady_clock::time_point current_time = std::chrono::steady_clock::now();
        std::chrono::steady_clock::time_point previous_time = current_time;

        update_canvas(canvas, cam_operator);
        txt_res->update(canvas);

        // Main loop
        while (!window.should_close()) {
            glfwPollEvents();

            // Start the Dear ImGui frame
            ImGui_ImplOpenGL3_NewFrame();
            ImGui_ImplGlfw_NewFrame();
            ImGui::NewFrame();
            {
                ImGui::Begin("Controls");
                ImGui::Button("Load Scene");
                ImGui::End();
            }

            ImGui::Render();

            // Calculate delta time
            current_time = std::chrono::steady_clock::now();
            std::chrono::duration<float> delta_time = std::chrono::duration_cast<std::chrono::duration<float>>(current_time - previous_time);
            previous_time = current_time;

            if (cam_operator.update(window, delta_time.count())) {
                update_canvas(canvas, cam_operator);
                txt_res->update(canvas);
            }

            // Update opengl viewport
            auto fb_size = window.get_framebuffer_size();
            glViewport(0, 0, fb_size.x, fb_size.y);

            // Clear framebuffer
            glClearColor(clear_color.x * clear_color.w, clear_color.y * clear_color.w, clear_color.z * clear_color.w, clear_color.w);
            glClear(GL_COLOR_BUFFER_BIT);

            sh.bind();
            txt.bind(0);
            billboard.draw();

            ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

            window.swap_buffers();
        }
        //Backend::terminate_imgui();
        // glfw window terminates here
    }
    Backend::terminate_glfw();

    return 0;
}
