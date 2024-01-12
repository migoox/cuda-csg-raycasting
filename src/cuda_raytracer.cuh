#ifndef CSG_RAY_TRACING_CUDA_RAYTRACER_CUH
#define CSG_RAY_TRACING_CUDA_RAYTRACER_CUH
#include <glm/glm.hpp>

namespace raytracer {
    using vec2 = glm::vec<2,float,glm::lowp>;
    using vec3 = glm::vec<3,float,glm::lowp>;
    using vec4 = glm::vec<4,float,glm::lowp>;
//
    using mat3 = glm::vec<3,float,glm::lowp>;
    using mat4 = glm::vec<4,float,glm::lowp>;

    uint32_t on_sphere_hit(vec3 hit_point, vec3 sphere_pos, float sphere_radius);

    uint32_t on_miss();

    uint32_t trace_ray(vec3 origin, vec3 dir);

    uint32_t per_pixel(int x, int y, vec2 canvas, vec3 eye, mat4 inv_proj, mat4 inv_view);
}


#endif //CSG_RAY_TRACING_CUDA_RAYTRACER_CUH
