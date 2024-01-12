#include "window.hpp"
#include "vendor/imgui/imgui.h"
#include "vendor/imgui/backend/imgui_impl_glfw.h"
#include "vendor/imgui/backend/imgui_impl_opengl3.h"
#include <iostream>
using namespace renderer;

std::string Backend::glsl_version;

static void glfw_error_callback(int error, const char* description) {
    std::cout << "[GLFW] Error" <<  error << description << std::endl;
}

void Backend::init_glfw() {
    glfwSetErrorCallback(glfw_error_callback);
    if (!glfwInit()) {
        std::cout << "[GLFW]: Failed to init glfw" << std::endl;
        std::terminate();
    }

    // Decide GL+GLSL versions
#if defined(IMGUI_IMPL_OPENGL_ES2)
    // GL ES 2.0 + GLSL 100
    glsl_version = "#version 100";
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 2);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
    glfwWindowHint(GLFW_CLIENT_API, GLFW_OPENGL_ES_API);
#elif defined(__APPLE__)
    // GL 3.2 + GLSL 150
    glsl_version = "#version 150";
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 2);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);  // 3.2+ only
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);            // Required on Mac
#else
    // GL 3.0 + GLSL 130
    glsl_version = "#version 130";
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);  // 3.2+ only
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);            // 3.0+ only
#endif
}

void Backend::init_glew() {
    // Initialize glew
    GLenum err = glewInit();
    if (GLEW_OK != err)
    {
        /* Problem: glewInit failed, something is seriously wrong. */
        std::cerr << "[GLEW] Error: " << glewGetErrorString(err) << std::endl;
        Backend::terminate_glfw();
        std::terminate();
    }

    std::cout << "[GLEW] Using GL Version: " << glGetString(GL_VERSION) << std::endl;
}

void Backend::init_imgui(Window &window) {
    // Setup Dear ImGui context
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO(); (void)io;
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;     // Enable Keyboard Controls
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;      // Enable Gamepad Controls

    // Setup Dear ImGui style
    ImGui::StyleColorsDark();

    // Setup Platform/Renderer backends
    ImGui_ImplGlfw_InitForOpenGL(window.get_raw(), true);
    ImGui_ImplOpenGL3_Init(glsl_version.c_str());
}


void Backend::terminate_imgui() {
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
}

void Backend::terminate_glfw() {
    glfwTerminate();
}

int Backend::to_glfw_mouse_btn_key(MouseButton mouse_button) {
    switch(mouse_button) {
        case MouseButton::Left:
            return GLFW_MOUSE_BUTTON_LEFT;
        case MouseButton::Middle:
            return GLFW_MOUSE_BUTTON_MIDDLE;
        case MouseButton::Right:
            return GLFW_MOUSE_BUTTON_RIGHT;
    }
}

void Window::glfw_on_win_resized(GLFWwindow *glfw_window, int width, int height) {
    auto window = static_cast<Window*>(glfwGetWindowUserPointer(glfw_window));
    window->m_size = { width, height } ;
}

void Window::glfw_on_mouse_moved(GLFWwindow *glfw_window, double xpos, double ypos) {
    auto window = static_cast<Window*>(glfwGetWindowUserPointer(glfw_window));
    window->m_mouse_pos = { static_cast<float>(xpos), static_cast<float>(ypos) };
}

void Window::glfw_on_win_moved(GLFWwindow *glfw_window, int x, int y) {
    auto window = static_cast<Window*>(glfwGetWindowUserPointer(glfw_window));
    window->m_pos = { x, y };
}

Window::Window(int width, int height, const char* title) : m_size({width, height}) {
    // Glfw window creation
    m_glfw_window = glfwCreateWindow(width, height, title, nullptr, nullptr);
    if (m_glfw_window == nullptr)
    {
        std::cout << "[GLFW]: Failed to create GLFW window" << std::endl;
        Backend::terminate_glfw();
        std::terminate();
    }

    // Attach opengl context
    glfwMakeContextCurrent(m_glfw_window);

    // Set callbacks
    glfwSetWindowSizeCallback(m_glfw_window, Window::glfw_on_win_resized);
    glfwSetWindowPosCallback(m_glfw_window, Window::glfw_on_win_moved);
    glfwSetCursorPosCallback(m_glfw_window, Window::glfw_on_mouse_moved);

    // Turn off vsync
    glfwSwapInterval(0);

    glfwSetWindowUserPointer(m_glfw_window, this);
}

Window::~Window() {
    glfwDestroyWindow(m_glfw_window);
}

Vec2i Window::get_framebuffer_size() const {
    int x, y;
    glfwGetFramebufferSize(m_glfw_window, &x, &y);
    return { x, y };
}

bool Window::is_key_pressed(int key) const {
    return glfwGetKey(m_glfw_window, key) == GLFW_PRESS;
}

bool Window::is_mouse_btn_pressed(MouseButton btn) const {
    return glfwGetMouseButton(m_glfw_window, Backend::to_glfw_mouse_btn_key(btn));
}

bool Window::should_close() const {
    return glfwWindowShouldClose(m_glfw_window);
}

void Window::swap_buffers() {
    glfwSwapBuffers(m_glfw_window);
}

void Window::set_cursor_mode(int value) const {
    glfwSetInputMode(m_glfw_window, GLFW_CURSOR, value);
}


