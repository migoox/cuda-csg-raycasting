#ifndef CSG_RAY_TRACING_CUDA_RAYTRACER_CUH
#define CSG_RAY_TRACING_CUDA_RAYTRACER_CUH
#include <cstdint>
#include "csg_utils.cuh"
#include "textures.hpp"
#include "camera_operator.hpp"

namespace cpu_raytracer {

    void update_canvas(renderer::Image& canvas, app::CameraOperator& cam_operator, const csg::CSGTree& tree, bool show_csg);

    float get_sphere_hit(glm::vec3 center, float radius, glm::vec3 ray_origin, glm::vec3 ray_dir, float min = 0.f);

    uint32_t on_miss();
    uint32_t on_hit(glm::vec3 hit_point, glm::vec3 normal, glm::vec3 color);

    uint32_t trace_ray(const csg::CSGTree& tree, const glm::vec3& origin, const glm::vec3& dir);
    uint32_t csg_trace_ray(const csg::CSGTree& tree, const glm::vec3& origin, const glm::vec3& dir);

    uint32_t per_pixel(int x, int y, glm::vec2 canvas, glm::vec3 eye,  glm::mat4 inv_proj,  glm::mat4 inv_view,
                       const csg::CSGTree& tree, bool csg = false);
}


#endif //CSG_RAY_TRACING_CUDA_RAYTRACER_CUH
