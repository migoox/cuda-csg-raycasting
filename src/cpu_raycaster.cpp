#define GLM_FORCE_CUDA
#include <glm/glm.hpp>
#include "cpu_raycaster.hpp"
#include "textures.hpp"
#include "cuda_runtime.h"
#include <cublas_v2.h>
#include <limits>

namespace cpu_raytracer {
    using namespace glm;

    // Returns parametric function argument (t)
    float get_sphere_hit(vec3 center, float radius, vec3 ray_origin, vec3 ray_dir) {
        vec3 co = ray_origin - center;
        // quadratic equation
        float a = glm::dot(ray_dir, ray_dir);
        float b = 2.f * glm::dot(ray_dir, co);
        float c = glm::dot(co, co) - radius * radius;

        float delta = b * b - 4 * a * c;

        if (delta < 0) {
            return -1.f;
        }

        return (-b - std::sqrt(delta)) / (2.f * a);
    }

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

    uint32_t trace_ray(vec3 origin, vec3 dir, const csg::CSGTree& tree) {
        float t_min = std::numeric_limits<float>::max();
        int closest_sphere = -1;
        for (int i = 0; i < tree.get_sphere_count(); ++i) {
            float t = get_sphere_hit(tree.sphere_center(i), tree.sphere_radius(i), origin, dir);

            if (t > 0.f && t < t_min) {
                t_min = t;
                closest_sphere = i;
            }
        }

        if (closest_sphere != -1) {
            vec3 closest_hit = origin + dir * t_min;
            // vec3 hit2 = origin + dir * (-b + std::sqrt(delta)) / (2.f * a);
            return on_sphere_hit(closest_hit, tree.sphere_center(closest_sphere),
                                 tree.sphere_radius(closest_sphere));
        }

        return on_miss();
    }

    uint32_t per_pixel(int x, int y, vec2 canvas, vec3 eye, mat4 inv_proj, mat4 inv_view, const csg::CSGTree& tree) {
        // Map pixel to nds coords with aspect ratio fix
        vec2 viewport_coords = { static_cast<float>(x) / canvas.x, (static_cast<float>(y)) / canvas.y };
        viewport_coords = viewport_coords * 2.0f - 1.0f;

        // Pixel's position in the world space
        vec4 target = inv_proj * vec4(viewport_coords.x, viewport_coords.y, -1.f, 1.f);
        vec3 dir = vec3(inv_view * vec4(normalize(vec3(target) / target.w), 0.f)); // World space

        return trace_ray(eye, dir, tree);
    }
}
