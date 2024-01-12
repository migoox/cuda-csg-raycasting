#ifndef OPENGL_3D_SCENERY_CAMERA_OPERATOR_H
#define OPENGL_3D_SCENERY_CAMERA_OPERATOR_H
#include "window.hpp"
#include "camera.hpp"

namespace app {
    class CameraOperator {
    public:
        CameraOperator() = delete;
        CameraOperator(float width, float height, float fov, float znear, float zfar);

        renderer::FPCamera& get_cam() { return m_cam; }
        bool update(const renderer::Window& window, float dt);

    private:
        glm::vec2 last_mouse_pos = glm::vec2(0.f, 0.f);
        renderer::FPCamera m_cam;
    };
}



#endif //OPENGL_3D_SCENERY_CAMERA_OPERATOR_H
