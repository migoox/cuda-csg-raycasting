#define GLM_FORCE_CUDA
#include <glm/glm.hpp>
#include "cpu_raytracer.h"
#include "textures.hpp"
#include "cuda_runtime.h"
#include <cublas_v2.h>


namespace cpu_raytracer {
    using namespace glm;

    uint32_t on_sphere_hit(vec3 hit_point, vec3 sphere_pos, float sphere_radius) {
        vec3 light_pos;
        light_pos.x = 0.f;
        light_pos.y = 1.f;
        light_pos.z = 0.f;

        vec3 normal = normalize(hit_point - sphere_pos);

        // normal = 0.5f * (normal + 1.f);
        // return get_color_rgb_norm(normal.r, normal.g, normal.b);

        vec3 light_dir = normalize(light_pos - hit_point);

        vec3 color = vec3(1.f, 0.f, 0.f) * glm::clamp(glm::dot(normal, light_dir), 0.f, 1.f);

        return renderer::get_color_rgb_norm(color.r, color.g, color.b);
    }

    uint32_t on_miss() {
        return renderer::get_color_rgb(60, 60, 60);
    }

    uint32_t trace_ray(vec3 origin, vec3 dir) {
        // Sphere
        vec3 sphere_pos = vec3(0.0f, 0.f, -1.0f); // world space position
        float sphere_radius = 0.5f;

        // -----------
        vec3 co = origin - sphere_pos;
        // quadratic equation
        float a = glm::dot(dir, dir);
        float b = 2.f * glm::dot(dir, co);
        float c = glm::dot(co, co) - sphere_radius * sphere_radius;

        float delta = b * b - 4 * a * c;
        if (delta > 0.f) {
            vec3 closest_hit = origin + dir * (-b - std::sqrt(delta)) / (2.f * a);
            // vec3 hit2 = origin + dir * (-b + std::sqrt(delta)) / (2.f * a);
            return on_sphere_hit(closest_hit, sphere_pos, sphere_radius);
        }
        return on_miss();
    }

    uint32_t per_pixel(int x, int y, vec2 canvas, vec3 eye, mat4 inv_proj, mat4 inv_view) {
        // Map pixel to nds coords with aspect ratio fix
//    auto viewport_coords = vec2(
//            (2.f * static_cast<float>(x) - canvas.x) / canvas.y,
//            (2.f * (canvas.y - static_cast<float>(y)) - canvas.y) / canvas.y
//    );
        vec2 viewport_coords = { static_cast<float>(x) / canvas.x, (static_cast<float>(y)) / canvas.y };
        viewport_coords = viewport_coords * 2.0f - 1.0f;

        // Pixel's position in the world space
        vec4 target = inv_proj * vec4(viewport_coords.x, viewport_coords.y, -1.f, 1.f);
        vec3 dir = vec3(inv_view * vec4(normalize(vec3(target) / target.w), 0.f)); // World space

        return trace_ray(eye, dir);
    }
}
