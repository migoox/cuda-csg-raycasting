#ifndef CSG_RAY_TRACING_CAMERA_H
#define CSG_RAY_TRACING_CAMERA_H
#include <glm/glm.hpp>
#include "GLFW/glfw3.h"

namespace camera {
// First person camera (right handed)
    class FPCamera {
    public:
        FPCamera() = delete;

        FPCamera(float width, float height, float fov, float znear, float zfar);

        void set_pos(glm::vec3 pos);

        glm::vec3 get_pos() const { return m_pos; }

        void move(float right, float up, float forward);

        // Rotates camera around the local x axis (pitch) and around the global y axis (yaw)
        // x and y should be in radians.
        void rotate(float x, float y);

        void resize_proj(float width, float height);

        void set_fov(float fov);

        void set_znear(float znear);

        void set_zfar(float zfar);

        const glm::mat4 &get_inverse_proj();

        const glm::mat4 &get_inverse_view();

    private:
        static constexpr glm::vec3 global_up = glm::vec3(0.f, 1.f, 0.f);

        void recalculate_proj();

        void recalculate_view();

    private:
        bool view_dirty = true;
        bool proj_dirty = true;

        float m_width, m_height, m_fov, m_znear, m_zfar;

        glm::vec3 m_forward = glm::vec3(0.f, 0.f, -1.f);
        glm::vec3 m_pos = glm::vec3(0.f, 0.f, 0.f);

        glm::mat4 m_inverse_proj = glm::mat4(1.f);
        glm::mat4 m_inverse_view = glm::mat4(1.f);
    };


}


#endif //CSG_RAY_TRACING_CAMERA_H
