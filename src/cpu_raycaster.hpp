#ifndef CSG_RAY_TRACING_CUDA_RAYTRACER_CUH
#define CSG_RAY_TRACING_CUDA_RAYTRACER_CUH
#include <cstdint>
#include "csg.hpp"

namespace cpu_raytracer {
    uint32_t on_sphere_hit(glm::vec3 hit_point, glm::vec3 sphere_pos, float sphere_radius);

    uint32_t on_miss();

    uint32_t trace_ray(glm::vec3 origin, glm::vec3 dir);

    uint32_t per_pixel(int x, int y, glm::vec2 canvas, glm::vec3 eye,  glm::mat4 inv_proj,  glm::mat4 inv_view, const csg::CSGTree& tree);
}


#endif //CSG_RAY_TRACING_CUDA_RAYTRACER_CUH
