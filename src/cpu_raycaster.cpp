#define GLM_FORCE_CUDA
#include <glm/glm.hpp>
#include "cpu_raycaster.hpp"
#include "textures.hpp"
#include "cuda_runtime.h"
#include <cublas_v2.h>
#include <limits>

namespace cpu_raytracer {
    using namespace glm;

    // Returns parametric function argument (t), returns -1.f in the case there was no hit
    float get_sphere_hit(vec3 center, float radius, vec3 ray_origin, vec3 ray_dir, float min) {
        vec3 co = ray_origin - center;

        // Quadratic equation
        float a = glm::dot(ray_dir, ray_dir);
        float b = 2.f * glm::dot(ray_dir, co);
        float c = glm::dot(co, co) - radius * radius;

        float delta = b * b - 4 * a * c;

        if (delta < 0.f) {
            return -1.f;
        }

        // The ray is in enter state for that sphere
        float t = (-b - std::sqrt(delta)) / (2.f * a);
        if (t > min) {
            return t;
        }

        // The ray is in exit state for that sphere
        t = (-b + std::sqrt(delta)) / (2.f * a);
        if (t > min) {
            return t;
        }

        // The ray has missed the sphere
        return -1.f;
    }

    uint32_t on_hit(vec3 hit_point, vec3 normal) {
        vec3 light_pos = vec3(0.f, 1.f, 0.f);


        // normal = 0.5f * (normal + 1.f);
        // return get_color_rgb_norm(normal.r, normal.g, normal.b);

        vec3 light_dir = normalize(light_pos - hit_point);

        vec3 color = vec3(1.f, 0.f, 0.f) * glm::clamp(glm::dot(normal, light_dir), 0.f, 1.f);

        return renderer::get_color_rgb_norm(color.r, color.g, color.b);
    }

    uint32_t on_miss() {
        // Background color
        return renderer::get_color_rgb(60, 60, 60);
    }

    enum class PointState {
        Enter,
        Exit,
        Miss
    };

    struct CSGActions {
        enum CSGAction {
            RetLeftIfCloser,
            RetRightIfCloser,
            LoopRight,
            LoopLeft,
            RetLeft,
            RetRight,
            LoopLeftIfCloser,
            LoopRightIfCloser,
            FlipRight,
            Miss,
            None
        };

        bool has_action(CSGAction action) {
            if (array[0] == action) {
                return true;
            }

            if (array[1] == action) {
                return true;
            }

            if (array[2] == action) {
                return true;
            }

            return false;
        }

        CSGActions(PointState state_l, PointState state_r, const csg::Node& node) {
            static CSGAction union_table[][3] = {
                    {RetLeftIfCloser, RetRightIfCloser, None},     {RetRightIfCloser, LoopLeft, None},          {RetLeft, None, None},
                    {RetLeftIfCloser, LoopRight, None},            {LoopLeftIfCloser, LoopRightIfCloser, None}, {RetLeft, None, None},
                    {RetRight, None, None},                        {RetRight, None, None},                      {Miss, None, None},
            };

            static CSGAction inter_table[][3] = {
                    {LoopLeftIfCloser, LoopRightIfCloser, None},   {RetLeftIfCloser, LoopRight, None},          {Miss, None, None},
                    {RetRightIfCloser, LoopLeft, None},            {RetLeftIfCloser, RetRightIfCloser, None},   {Miss, None, None},
                    {Miss, None, None},                            {Miss, None, None},                          {Miss, None, None},
            };

            static CSGAction diff_table[][3] = {
                    {RetLeftIfCloser, LoopRight, None},            {LoopLeftIfCloser, LoopRightIfCloser, None}, {RetLeft, None, None},
                    {RetLeftIfCloser, RetRightIfCloser, FlipRight},{RetRightIfCloser, FlipRight, LoopLeft},     {RetLeft, None, None},
                    {Miss, None, None},                            {Miss, None, None},                          {Miss, None, None},
            };

            if (node.type == csg::Node::Type::UnionOp) {
                array[0] = union_table[(int)state_l * 3 + (int)state_r][0];
                array[1] = union_table[(int)state_l * 3 + (int)state_r][1];
                array[2] = union_table[(int)state_l * 3 + (int)state_r][2];

            } else if (node.type == csg::Node::Type::DiffOp) {
                array[0] = diff_table[(int)state_l * 3 + (int)state_r][0];
                array[1] = diff_table[(int)state_l * 3 + (int)state_r][1];
                array[2] = diff_table[(int)state_l * 3 + (int)state_r][2];
            } else if (node.type == csg::Node::Type::InterOp) {
                array[0] = inter_table[(int)state_l * 3 + (int)state_r][0];
                array[1] = inter_table[(int)state_l * 3 + (int)state_r][1];
                array[2] = inter_table[(int)state_l * 3 + (int)state_r][2];
            }
        }

        CSGAction array[3] = {None, None, None };
    };

    PointState csg_point_classify(float t, vec3 normal, vec3 ray_dir) {
        if (t == 0.f) {
            return PointState::Miss;
        }

        if (dot(normal, ray_dir) > 0.f) {
            return PointState::Exit;
        }

        if (dot(normal, ray_dir) < 0.f) {
            return PointState::Enter;
        }

        return PointState::Miss;
    }

    struct IntersectionResult {
        float t; // if t == -1.0f => miss
        vec3 normal;
    };

    IntersectionResult csg_intersect(const csg::CSGTree& tree, const vec3& origin, const vec3& dir, csg::Node node, float min) {
        // Stop condition
        if (node.type == csg::Node::Type::Sphere) {
            float t = get_sphere_hit(tree.sphere_center(node.prim_id), tree.sphere_radius(node.prim_id), origin, dir, min);
            return IntersectionResult {
                    t,
                    t == -1.f ? vec3(0.f) : normalize(origin + t * dir - tree.sphere_center(node.prim_id))
            };
        }

        float min_l = min;
        float min_r = min;

        // Recursive call
        auto res_l = csg_intersect(tree, origin, dir, tree.get_node(node.get_left_id()), min_l);
        auto res_r = csg_intersect(tree, origin, dir, tree.get_node(node.get_right_id()), min_r);

        auto state_l = csg_point_classify(res_l.t, res_l.normal, dir);
        auto state_r = csg_point_classify(res_r.t, res_r.normal, dir);
        while (true) {
            CSGActions actions = CSGActions(state_l, state_r, node);
            if (actions.has_action(CSGActions::Miss)) {
                return IntersectionResult { -1.f, vec3(0.f) }; // Miss
            }

            if (actions.has_action(CSGActions::RetLeft) ||
                (actions.has_action(CSGActions::RetLeftIfCloser) && res_l.t <= res_r.t)) {
                return res_l;
            }

            if (actions.has_action(CSGActions::RetRight) ||
                (actions.has_action(CSGActions::RetRightIfCloser) && res_r.t <= res_l.t)) {
                if (actions.has_action(CSGActions::FlipRight)) {
                    return IntersectionResult { res_r.t, -res_r.normal };
                }
                return res_r;
            }

            if (actions.has_action(CSGActions::LoopLeft) ||
                    (actions.has_action(CSGActions::LoopLeftIfCloser) && res_l.t <= res_r.t)) {
                min_l = res_l.t;
                res_l = csg_intersect(tree, origin, dir, tree.get_node(node.get_left_id()), min_l);
                state_l = csg_point_classify(res_l.t, res_l.normal, dir);
            } else if (actions.has_action(CSGActions::LoopRight) ||
                    (actions.has_action(CSGActions::LoopRightIfCloser) && res_r.t <= res_l.t)) {
                min_r = res_r.t;
                res_r = csg_intersect(tree, origin, dir, tree.get_node(node.get_right_id()), min_r);
                state_r = csg_point_classify(res_r.t, res_r.normal, dir);
            } else {
                return IntersectionResult { -1.f, vec3(0.f) }; // Miss
            }
        }
    }

    uint32_t csg_trace_ray(const csg::CSGTree& tree, const vec3& origin, const vec3& dir) {
        if (tree.get_nodes_count() <= 1) {
            return on_miss();
        }

        auto result = csg_intersect(tree, origin, dir, tree.get_node(1), 0.f);

        if (result.t == -1.0f) {
            return on_miss();
        }

        return on_hit(origin + dir * result.t, result.normal);
    }

    uint32_t trace_ray(const csg::CSGTree& tree, const vec3& origin, const vec3& dir) {
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
            return on_hit(closest_hit, normalize(closest_hit - tree.sphere_center(closest_sphere)));
        }

        return on_miss();
    }

    uint32_t per_pixel(int x, int y, vec2 canvas, vec3 eye, mat4 inv_proj, mat4 inv_view, const csg::CSGTree& tree, bool csg) {
        vec2 viewport_coords = { static_cast<float>(x) / canvas.x, (static_cast<float>(y)) / canvas.y };
        viewport_coords = viewport_coords * 2.0f - 1.0f;

        // pixel's position in the world space
        vec4 target = inv_proj * vec4(viewport_coords.x, viewport_coords.y, -1.f, 1.f);
        vec3 dir = vec3(inv_view * vec4(normalize(vec3(target) / target.w), 0.f)); // world space

        if (csg) {
            return csg_trace_ray(tree, eye, dir);
        } else {
            return trace_ray(tree, eye, dir);
        }
    }


}
