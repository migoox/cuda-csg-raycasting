#include "window.hpp"
#include <stdio.h>
#include <exception>
#include "imgui_internal.h"
#include "imgui.h"
#include "vendor/imgui/backend/imgui_impl_glfw.h"
#include "vendor/imgui/backend/imgui_impl_opengl3.h"

uint32_t Window::s_width, Window::s_height;
uint32_t Window::s_pos_x, Window::s_pos_y;

ImVec2 Window::s_old_size;
ImVec2 Window::s_old_pos;
bool Window::s_is_fullscreen;

GLFWwindow* Window::s_win_handle;

void Window::init(uint32_t width, uint32_t height) {
    s_old_pos = ImVec2(0, 0);
    s_old_size = ImVec2(1280, 720);

    s_is_fullscreen = false;
    s_width = width;
    s_height = height;

    glfwSetErrorCallback(Window::on_glfw_error);
    if (!glfwInit())
        std::terminate();

    // Decide GL+GLSL versions
#if defined(IMGUI_IMPL_OPENGL_ES2)
    // GL ES 2.0 + GLSL 100
    const char* glsl_version = "#version 100";
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 2);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
    glfwWindowHint(GLFW_CLIENT_API, GLFW_OPENGL_ES_API);
#elif defined(__APPLE__)
    // GL 3.2 + GLSL 150
    const char* glsl_version = "#version 150";
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 2);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);  // 3.2+ only
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);            // Required on Mac
#else
    // GL 3.0 + GLSL 130
    const char* glsl_version = "#version 130";
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
    //glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);  // 3.2+ only
    //glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);            // 3.0+ only
#endif

    // Create window with graphics context
    s_win_handle = glfwCreateWindow(1280, 720, "CSG Ray Tracing", nullptr, nullptr);
    if (s_win_handle == nullptr)
        std::terminate();

    glfwMakeContextCurrent(s_win_handle);
    glfwSwapInterval(0);
    glfwSetWindowSizeCallback(s_win_handle, Window::on_resize);
    glfwSetWindowPosCallback(s_win_handle, Window::on_pos);


    // Setup Dear ImGui context
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO(); (void)io;
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;     // Enable Keyboard Controls
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;      // Enable Gamepad Controls

    // Setup Dear ImGui style
    ImGui::StyleColorsDark();

    // Setup Platform/Renderer backends
    ImGui_ImplGlfw_InitForOpenGL(Window::get_win_handle(), true);


#ifdef __EMSCRIPTEN__
    ImGui_ImplGlfw_InstallEmscriptenCanvasResizeCallback("#canvas");
#endif
    ImGui_ImplOpenGL3_Init(glsl_version);

}

void Window::on_pos(GLFWwindow* window, int x, int y) {
    s_pos_x = x;
    s_pos_y = y;
}

void Window::on_resize(GLFWwindow* window, int width, int height) {
    s_height = height;
    s_width = width;
}

void Window::on_glfw_error(int error, const char *description) {
    fprintf(stderr, "[GLFW] Error %d: %s\n", error, description);
}

std::vector<std::pair<int, int>> Window::get_supported_resolutions() {
    std::vector<std::pair<int, int>> resolutions;
    resolutions.emplace_back(1920, 1080);
    resolutions.emplace_back(1280, 720);
    resolutions.emplace_back(1024, 768);

    return resolutions;
}

void Window::toggle_fullscreen() {
    s_is_fullscreen = !s_is_fullscreen;

    if (s_is_fullscreen) {
        s_old_size = ImVec2(s_width, s_height);
        s_old_pos = ImVec2(s_pos_x, s_pos_y);
        GLFWmonitor* monitor = glfwGetPrimaryMonitor();
        const GLFWvidmode* mode = glfwGetVideoMode(monitor);
        glfwSetWindowMonitor(s_win_handle, monitor, 0, 0, mode->width, mode->height, GLFW_DONT_CARE);
    } else {
        glfwSetWindowMonitor(s_win_handle, nullptr, s_old_pos.x, s_old_pos.y, s_old_size.x, s_old_size.y, GLFW_DONT_CARE);
    }
}

std::pair<int, int> Window::get_framebuffer_size() {
    int x, y;
    glfwGetFramebufferSize(s_win_handle, &x, &y);
    return std::pair<int, int>(x, y);
}

void Window::swap_buffers() {
    glfwSwapBuffers(s_win_handle);
}

void Window::destroy() {
    // Cleanup
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    glfwDestroyWindow(s_win_handle);
    glfwTerminate();
}

GLFWwindow *Window::get_win_handle() {
     return s_win_handle;
}



