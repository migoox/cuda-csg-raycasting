#include "camera.h"
#include <cmath>
#include <glm/ext/matrix_clip_space.hpp>
#include <glm/gtx/transform.hpp>

using namespace camera;

FPCamera::FPCamera(float width, float height, float fov, float znear, float zfar)
: m_width(width), m_height(height), m_fov(fov), m_znear(znear), m_zfar(zfar) {

}

void FPCamera::move(float right, float up, float forward) {
    if (right == 0.f && up == 0.f && forward == 0.f) {
        return;
    }

    view_dirty = true;
    m_pos += global_up * up;
    m_pos += m_forward * forward;

    if (right == 0.f) {
        return;
    }
    glm::vec3 right_dir = glm::normalize(glm::cross(m_forward, global_up));
    m_pos += right_dir * right;
}

void FPCamera::rotate(float x, float y) {
    if (x == 0.f && y == 0.f) {
        return;
    }

    view_dirty = true;
    m_forward = glm::vec3(glm::rotate(y, global_up) * glm::vec4(m_forward, 0.f));

    if (x == 0.f) {
        return;
    }

    glm::vec3 right_dir = glm::normalize(glm::cross(m_forward, global_up));
    m_forward = glm::vec3(glm::rotate(x, right_dir) * glm::vec4(m_forward, 0.f));
}

const glm::mat4 &FPCamera::get_inverse_proj() {
    this->recalculate_proj();

    return m_inverse_proj;
}

const glm::mat4 &FPCamera::get_inverse_view() {
    this->recalculate_view();

    return m_inverse_view;
}

void FPCamera::recalculate_proj() {
    if (!proj_dirty) {
        return;
    }
    proj_dirty = false;
    m_fov = glm::radians(45.f);

    float tan_half_fov = glm::tan(m_fov / 2.f);
    float aspect = m_width / m_height;
    m_inverse_proj = glm::mat4(0.f);
    m_inverse_proj[0][0] = aspect * tan_half_fov;
    m_inverse_proj[1][1] = tan_half_fov;
    m_inverse_proj[2][3] = (m_znear - m_zfar) / (2.f * m_zfar * m_znear);
    m_inverse_proj[3][2] = -1.f;
    m_inverse_proj[3][3] = (m_znear + m_zfar) / (2.f * m_zfar * m_znear);

    //m_inverse_proj = glm::inverse(glm::perspective(glm::radians(45.f), m_width / m_height, m_znear, m_zfar));
}

void FPCamera::recalculate_view() {
    if (!view_dirty) {
        return;
    }
    view_dirty = false;

    // Assuming rh coordinate system
    glm::vec3 const right(glm::normalize(glm::cross(m_forward, global_up)));
    glm::vec3 const up(glm::normalize(glm::cross(right, m_forward)));

    m_inverse_view = glm::mat4(1.f);
    m_inverse_view[0][0] = right.x;
    m_inverse_view[0][1] = right.y;
    m_inverse_view[0][2] = right.z;

    m_inverse_view[1][0] = up.x;
    m_inverse_view[1][1] = up.y;
    m_inverse_view[1][2] = up.z;

    m_inverse_view[2][0] = -m_forward.x;
    m_inverse_view[2][1] = -m_forward.y;
    m_inverse_view[2][2] = -m_forward.z;

    m_inverse_view[3][0] = m_pos.x;
    m_inverse_view[3][1] = m_pos.y;
    m_inverse_view[3][2] = m_pos.z;
    m_inverse_view[3][3] = 1.f;

    //m_inverse_view = glm::inverse(glm::lookAtRH(m_pos, m_pos + m_forward, global_up));
}

void FPCamera::set_pos(glm::vec3 pos) {
    m_pos = pos;
}

void FPCamera::resize_proj(float width, float height) {
    proj_dirty = true;
    m_width = width;
    m_height = height;
}

void FPCamera::set_fov(float fov) {
    proj_dirty = true;
    m_fov = fov;
}

void FPCamera::set_znear(float znear) {
    proj_dirty = true;
    m_znear = znear;
}

void FPCamera::set_zfar(float zfar) {
    proj_dirty = true;
    m_zfar = zfar;
}


