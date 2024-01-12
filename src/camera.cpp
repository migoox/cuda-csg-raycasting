#include "camera.hpp"
#include <cmath>
#include "glm/ext/matrix_clip_space.hpp"
#include "glm/gtx/transform.hpp"
#include <algorithm>

using namespace renderer;

FPCamera::FPCamera(float width, float height, float fov, float znear, float zfar)
: PerspectiveProjection(width, height, fov, znear, zfar) { }

void FPCamera::move(float right, float up, float forward) {
    if (right == 0.f && up == 0.f && forward == 0.f) {
        return;
    }

    view_dirty = true;
    m_pos += global_up() * up;
    m_pos += m_forward * forward;

    if (right == 0.f) {
        return;
    }
    glm::vec3 right_dir = glm::normalize(glm::cross(m_forward, global_up()));
    m_pos += right_dir * right;
}

void FPCamera::rotate(float x, float y) {
    if (x == 0.f && y == 0.f) {
        return;
    }

    view_dirty = true;
    m_forward = glm::vec3(glm::rotate(y, global_up()) * glm::vec4(m_forward, 0.f));

    if (x == 0.f) {
        return;
    }

    glm::vec3 right_dir = glm::normalize(glm::cross(m_forward, global_up()));
    m_forward = glm::vec3(glm::rotate(x, right_dir) * glm::vec4(m_forward, 0.f));
}

const glm::mat4 &FPCamera::get_inv_view() {
    this->recalculate_view();

    return m_inv_view;
}


const glm::mat4 &FPCamera::get_view() {
    this->recalculate_view();

    return m_view;
}

void FPCamera::recalculate_view() {
    if (!view_dirty) {
        return;
    }
    view_dirty = false;

    // Assuming rh coordinate system
    glm::vec3 const right(glm::normalize(glm::cross(m_forward, global_up())));
    glm::vec3 const up(glm::normalize(glm::cross(right, m_forward)));

    m_inv_view = glm::mat4(1.f);
    m_inv_view[0][0] = right.x;
    m_inv_view[0][1] = right.y;
    m_inv_view[0][2] = right.z;

    m_inv_view[1][0] = up.x;
    m_inv_view[1][1] = up.y;
    m_inv_view[1][2] = up.z;

    m_inv_view[2][0] = -m_forward.x;
    m_inv_view[2][1] = -m_forward.y;
    m_inv_view[2][2] = -m_forward.z;

    m_inv_view[3][0] = m_pos.x;
    m_inv_view[3][1] = m_pos.y;
    m_inv_view[3][2] = m_pos.z;

    m_view = glm::mat4(1.f);
    m_view[0][0] = right.x;
    m_view[1][0] = right.y;
    m_view[2][0] = right.z;

    m_view[0][1] = up.x;
    m_view[1][1] = up.y;
    m_view[2][1] = up.z;

    m_view[0][2] =-m_forward.x;
    m_view[1][2] =-m_forward.y;
    m_view[2][2] =-m_forward.z;

    m_view[3][0] =-dot(right, m_pos);
    m_view[3][1] =-dot(up, m_pos);
    m_view[3][2] = dot(m_forward, m_pos);
}

void FPCamera::set_pos(glm::vec3 pos) {
    m_pos = pos;
}


const glm::mat4 &OrbitingCamera::get_view() const {
    this->recalculate_view();
    return m_view_mat;
}

OrbitingCamera::OrbitingCamera(float width, float height, float fov, float znear, float zfar)
: PerspectiveProjection(width, height, fov, znear, zfar) { }

void OrbitingCamera::recalculate_view() const {
    if (!m_view_dirty) {
        return;
    }
    m_view_dirty = false;

    // Convert camera position described in spherical coordinates to the cartesian coordinates
    auto eye = glm::vec3(
            m_radius * std::sin(m_polar_angle) * std::cos(m_azimuthal_angle) + m_center.x,
            m_radius * std::cos(m_polar_angle) + m_center.y,
            m_radius * std::sin(m_polar_angle) * std::sin(m_azimuthal_angle) + m_center.z
    );

    if (m_polar_angle <= 1e-4) {
        // Fix look at matrix (default up vector is invalid in this case)
        m_view_mat = glm::lookAtRH(
                eye,
                m_center,
                glm::vec3(
                        glm::rotate(
                                glm::mat4(1.0f),
                                glm::pi<float>() / 2.0f - m_azimuthal_angle,
                                glm::vec3(0.0f, 1.0f, 0.0f)
                        )
                        * glm::vec4(0.0, 0.0, -1.0, 1.0)
                )
        );
        return;
    }

    m_view_mat = glm::lookAtRH(
            eye,
            m_center,
            glm::vec3(0.0, 1.0, 0.0)
    );
}

void OrbitingCamera::update_radius(float delta) {
    if (m_radius + delta >= 0.5f) {
        m_view_dirty = true;
        m_radius += delta;
    }
}

void OrbitingCamera::update_polar_angle(float delta) {
    m_view_dirty = true;
    m_polar_angle = std::clamp(m_polar_angle + delta, 0.0f, 0.8f * glm::pi<float>());
}

void OrbitingCamera::update_azimuthal_angle(float delta) {
    m_view_dirty = true;
    m_azimuthal_angle += delta;
}

void OrbitingCamera::set_center(glm::vec3 center) {
    m_view_dirty = true;
    m_center = center;
}

PerspectiveProjection::PerspectiveProjection(float width, float height, float fov, float znear, float zfar)
: m_width(width), m_height(height), m_fov(fov), m_znear(znear), m_zfar(zfar)  { }

const glm::mat4 &PerspectiveProjection::get_inv_proj() const {
    this->recalculate();

    return m_inv_proj;
}

const glm::mat4 &PerspectiveProjection::get_proj() const {
    this->recalculate();

   return m_proj;
}

void PerspectiveProjection::resize_screen(float width, float height) {
    m_dirty = true;
    m_width = width;
    m_height = height;
}

void PerspectiveProjection::set_fov(float fov) {
    m_dirty = true;
    m_fov = fov;
}

void PerspectiveProjection::set_znear(float znear) {
    m_dirty = true;
    m_znear = znear;
}

void PerspectiveProjection::set_zfar(float zfar) {
    m_dirty = true;
    m_zfar = zfar;
}

void PerspectiveProjection::recalculate() const {
    if (!m_dirty) {
        return;
    }
    m_dirty = false;

    float tan_half_fov = glm::tan(m_fov / 2.f);
    float aspect = m_width / m_height;
    m_inv_proj = glm::mat4(0.f);
    m_inv_proj[0][0] = aspect * tan_half_fov;
    m_inv_proj[1][1] = tan_half_fov;
    m_inv_proj[2][3] = (m_znear - m_zfar) / (2.f * m_zfar * m_znear);
    m_inv_proj[3][2] = -1.f;
    m_inv_proj[3][3] = (m_znear + m_zfar) / (2.f * m_zfar * m_znear);

    m_proj = glm::perspective(m_fov, m_width / m_height, m_znear, m_zfar);
}