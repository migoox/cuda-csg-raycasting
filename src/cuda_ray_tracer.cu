#include "cuda_ray_tracer.cuh"
#include "image.hpp"
#include <stdio.h>
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
    glm::vec3 sphere_pos = glm::vec3(0.0f, 0.f, -1.0f); // world space position
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

uint32_t per_pixel(int x, int y, glm::vec2 canvas, glm::vec3 eye, glm::mat4 inv_proj, glm::mat4 inv_view) {
    // Map pixel to nds coords with aspect ratio fix
//    auto viewport_coords = glm::vec2(
//            (2.f * static_cast<float>(x) - canvas.x) / canvas.y,
//            (2.f * (canvas.y - static_cast<float>(y)) - canvas.y) / canvas.y
//    );
    glm::vec2 viewport_coords = { static_cast<float>(x) / canvas.x, (canvas.y - static_cast<float>(y)) / canvas.y };
    viewport_coords = viewport_coords * 2.0f - 1.0f;

    // Pixel's position in the world space
    glm::vec4 target = inv_proj * glm::vec4(viewport_coords.x, viewport_coords.y, -1.f, 1.f);
    glm::vec3 dir = glm::vec3(inv_view * glm::vec4(glm::normalize(glm::vec3(target) / target.w), 0.f)); // World space

    return trace_ray(eye, dir);
}

