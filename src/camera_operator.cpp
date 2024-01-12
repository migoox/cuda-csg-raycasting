#include "camera_operator.hpp"

app::CameraOperator::CameraOperator(float width, float height, float fov, float znear, float zfar)
: m_cam(width, height, fov, znear, zfar) { }

bool app::CameraOperator::update(const renderer::Window &window, float dt) {
    const float sensitivity = 0.06f;
    const float speed = 5.0f;

    glm::vec2 mouse_pos = glm::vec2(window.get_mouse_pos().x, window.get_mouse_pos().y);
    glm::vec2 mouse_delta = (mouse_pos - last_mouse_pos) * sensitivity;
    last_mouse_pos = mouse_pos;

    if (!window.is_mouse_btn_pressed(renderer::MouseButton::Right)) {
        window.set_cursor_mode(GLFW_CURSOR_NORMAL);
        return false;
    }
    window.set_cursor_mode(GLFW_CURSOR_DISABLED);

    float r = 0.f, u = 0.f, f = 0.f;

    if (window.is_key_pressed(GLFW_KEY_W)) {
        f += speed * dt;
    }
    else if (window.is_key_pressed(GLFW_KEY_S)) {
        f -= speed * dt;
    }

    if (window.is_key_pressed(GLFW_KEY_A)) {
        r -= speed * dt;
    }
    else if (window.is_key_pressed(GLFW_KEY_D)) {
        r += speed * dt;
    }

    if (window.is_key_pressed(GLFW_KEY_Q)) {
        u -= speed * dt;
    }
    else if (window.is_key_pressed(GLFW_KEY_E)) {
        u += speed * dt;
    }

    m_cam.rotate(-mouse_delta.y * 0.03f, -mouse_delta.x * 0.03f);
    m_cam.move(r, u, f);
    return true;
}
