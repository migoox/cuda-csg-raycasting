#ifndef CSG_RAY_TRACING_WINDOW_H
#define CSG_RAY_TRACING_WINDOW_H

#include <cstdint>
#include "GLFW/glfw3.h"
#include "imgui.h"
#include <vector>

class Window {

public:
    static void init(uint32_t width, uint32_t height);
    static void destroy();

    static GLFWwindow* get_win_handle();

    static bool should_close() { return glfwWindowShouldClose(s_win_handle); }
    static bool is_fullscreen() { return s_is_fullscreen; }
    static void toggle_fullscreen();

    static uint32_t get_width() { return s_width; }
    static uint32_t get_height() { return s_height; }

    static uint32_t get_pos_x() { return s_pos_x; }
    static uint32_t get_pos_y() { return s_pos_y; }

    static std::pair<int, int> get_framebuffer_size();

    static void swap_buffers();

private:
    static void on_resize(GLFWwindow* window, int width, int height);
    static void on_pos(GLFWwindow* window, int x, int y);
    static void on_glfw_error(int error, const char* description);


private:
    static uint32_t s_width, s_height;
    static uint32_t s_pos_x, s_pos_y;

    static ImVec2 s_old_size;
    static ImVec2 s_old_pos;
    static bool s_is_fullscreen;

    static GLFWwindow* s_win_handle;
};


#endif //CSG_RAY_TRACING_WINDOW_H
