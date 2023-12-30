#include "cuda_ray_tracer.cuh"
#include "image.hpp"

uint32_t on_sphere_hit(glm::vec3 hit_point, glm::vec3 sphere_pos, float sphere_radius) {
    glm::vec3 light_pos = glm::vec3(0.f, 1.f, 0.f);

    glm::vec3 normal = glm::normalize(hit_point - sphere_pos);

    // normal = 0.5f * (normal + 1.f);
    // return img::get_color_rgb_norm(normal.r, normal.g, normal.b);

    glm::vec3 light_dir = glm::normalize(light_pos - hit_point);

    glm::vec3 color = glm::vec3(1.f, 0.f, 0.f) * glm::clamp(glm::dot(normal, light_dir), 0.f, 1.f);

    return img::get_color_rgb_norm(color.r, color.g, color.b);
}

uint32_t on_miss() {
    return img::get_color_rgb(60, 60, 60);
}

uint32_t trace_ray(glm::vec3 origin, glm::vec3 dir) {
    // Sphere
    glm::vec3 sphere_pos = glm::vec3(0.0f, 0.f, -1.f); // world space position
    float sphere_radius = 0.5f;

    // -----------
    glm::vec3 co = origin - sphere_pos;
    // quadratic equation
    float a = glm::dot(dir, dir);
    float b = 2.f * glm::dot(dir, co);
    float c = glm::dot(co, co) - sphere_radius * sphere_radius;

    float delta = b * b - 4 * a * c;
    if (delta > 0.f) {
        glm::vec3 closest_hit = origin + dir * (-b - std::sqrt(delta)) / (2.f * a);
        // glm::vec3 hit2 = origin + dir * (-b + std::sqrt(delta)) / (2.f * a);
        return on_sphere_hit(closest_hit, sphere_pos, sphere_radius);
    }
    return on_miss();
}

uint32_t per_pixel(glm::vec2 viewport_coords) {
    // Camera
    glm::vec3 eye = glm::vec3(0.f, 0.f, 0.f); // world space camera position
    float focal_length = 1.f; // alpha/2 = 45 degrees

    // -----------------
    // Pixel's position in the camera space
    glm::vec3 pixel_pos = glm::vec3(viewport_coords.x, viewport_coords.y, -focal_length);

    // For now camera space == world space, so the direction is
    glm::vec3 dir = glm::normalize(pixel_pos - eye);
    return trace_ray(eye, dir);
}

