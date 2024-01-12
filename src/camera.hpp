#ifndef OPENGL_3D_SCENERY_CAMERA_HPP
#define OPENGL_3D_SCENERY_CAMERA_HPP

#include <glm/glm.hpp>
#include <GLFW/glfw3.h>

namespace renderer {
    static glm::vec3 global_up() { return glm::vec3(0.f, 1.f, 0.f); }

    class PerspectiveProjection {
    public:
        PerspectiveProjection() = delete;
        PerspectiveProjection(float width, float height, float fov, float znear, float zfar);

        void resize_screen(float width, float height);
        void set_fov(float fov);
        void set_znear(float znear);
        void set_zfar(float zfar);

        const glm::mat4& get_proj() const;
        const glm::mat4& get_inv_proj() const;

    private:
        void recalculate() const;

    private:
        mutable bool m_dirty = true;

        mutable glm::mat4 m_inv_proj = glm::mat4(1.f);
        mutable glm::mat4 m_proj = glm::mat4(1.f);
        float m_width, m_height, m_fov, m_znear, m_zfar;
    };

    // First person camera (right handed)
    class FPCamera : public PerspectiveProjection {
    public:
        FPCamera() = delete;
        FPCamera(float width, float height, float fov, float znear, float zfar);

        void set_pos(glm::vec3 pos);

        glm::vec3 get_pos() const { return m_pos; }

        void move(float right, float up, float forward);

        // Rotates camera around the local x axis (pitch) and around the global y axis (yaw)
        // x and y should be in radians.
        void rotate(float x, float y);

        const glm::mat4 &get_inv_view();
        const glm::mat4 &get_view();

    private:
        void recalculate_view();

    private:
        bool view_dirty = true;

        glm::vec3 m_forward = glm::vec3(0.f, 0.f, -1.f);
        glm::vec3 m_pos = glm::vec3(0.f, 0.f, 0.f);

        glm::mat4 m_inv_view = glm::mat4(1.f);
        glm::mat4 m_view = glm::mat4(1.f);
    };

    class OrbitingCamera : public PerspectiveProjection {
    public:
        OrbitingCamera() = delete;

        OrbitingCamera(float width, float height, float fov, float znear, float zfar);

        const glm::mat4& get_view() const;

        void set_center(glm::vec3 center);

        void update_radius(float delta);

        // Angles should be in radians
        void update_polar_angle(float delta);

        // Angles should be in radians
        void update_azimuthal_angle(float delta);

    private:
        void recalculate_view() const;

    private:
        mutable bool m_view_dirty = true;

        mutable glm::mat4 m_view_mat = glm::mat4(1.f);

        glm::vec3 m_center = glm::vec3(0.f);

        float m_radius = 1.f;
        float m_polar_angle = 0.f;
        float m_azimuthal_angle = 0.f;
    };
}

#endif
