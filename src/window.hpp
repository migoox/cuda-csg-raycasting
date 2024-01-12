#ifndef OPENGL_3D_SCENERY_WINDOW_HPP
#define OPENGL_3D_SCENERY_WINDOW_HPP

#include <string>
#include <GL/glew.h>
#include <GLFW/glfw3.h>

namespace renderer {
    struct Vec2i {
        int x, y;
    };
    struct Vec2f {
        float x, y;
    };

    enum class MouseButton {
        Left,
        Middle,
        Right
    };

    class Window;

    class Backend {
    public:
        static void init_imgui(Window& window);
        static void terminate_imgui();

        static void init_glfw();
        static void terminate_glfw();
        static int to_glfw_mouse_btn_key(MouseButton mouse_button);

        static void init_glew();
    private:
        static std::string glsl_version;
    };

    class Window {
    public:
        Window() = delete;
        Window(int width, int height, const char* title);
        ~Window();

        const Vec2i& get_pos() const { return m_pos; }
        const Vec2i& get_size() const { return m_size; }
        const Vec2f& get_mouse_pos() const { return m_mouse_pos; }
        Vec2i get_framebuffer_size() const;

        bool is_key_pressed(int key) const;
        bool is_mouse_btn_pressed(MouseButton btn) const;

        void set_cursor_mode(int value) const;

        GLFWwindow *get_raw() { return m_glfw_window; }

        bool should_close() const;
        void swap_buffers();

    private:
        static void glfw_on_win_resized(GLFWwindow *glfw_window, int width, int height);
        static void glfw_on_win_moved(GLFWwindow *glfw_window, int x, int y);
        static void glfw_on_mouse_moved(GLFWwindow *glfw_window, double xpos, double ypos);

    private:
        GLFWwindow* m_glfw_window;

        Vec2i m_size;
        Vec2i m_pos = { 0, 0 };

        Vec2f m_mouse_pos = { 0.f, 0.f };
    };
}

#endif //OPENGL_3D_SCENERY_WINDOW_HPP
