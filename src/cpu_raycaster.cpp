#include <glm/glm.hpp>
#include "cpu_raycaster.hpp"
#include "textures.hpp"
#include <limits>

namespace cpu_raytracer {
    using namespace glm;
    using namespace csg;

    void update_canvas(renderer::Image& canvas, app::CameraOperator& cam_operator, const csg::CSGTree& tree, bool show_csg) {
        for (int y = 0; y < canvas.get_height(); y++) {
            for (int x = 0; x < canvas.get_width(); x++) {
                canvas.set_pixel(x, y, cpu_raytracer::per_pixel(x, y,
                                                                glm::vec2(canvas.get_width(), canvas.get_height()),
                                                                cam_operator.get_cam().get_pos(),
                                                                cam_operator.get_cam().get_inv_proj(),
                                                                cam_operator.get_cam().get_inv_view(),
                                                                tree,
                                                                show_csg));
            }
        }
    }

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

    uint32_t on_hit(vec3 hit_point, vec3 normal, vec3 color) {
        vec3 light_pos = vec3(0.f, 1.f, 0.f);

        // normal = 0.5f * (normal + 1.f);
        // return get_color_rgb_norm(normal.r, normal.g, normal.b);

        vec3 light_dir = normalize(light_pos - hit_point);

        vec3 res_color = color * glm::clamp(glm::dot(normal, light_dir), 0.f, 1.f);

        return renderer::get_color_rgb_norm(res_color.r, res_color.g, res_color.b);
    }

    uint32_t on_miss() {
        // Background color
        return renderer::get_color_rgb(60, 60, 60);
    }

    csg::PointState csg_point_classify(float t, glm::vec3 normal, glm::vec3 ray_dir) {
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

    csg::IntersectionResult csg_intersect(const csg::CSGTree& tree, const vec3& origin, const vec3& dir, csg::Node node, float min) {
        // Stop condition
        if (node.type == csg::Node::Type::Sphere) {
            float t = get_sphere_hit(tree.sphere_center(node.context_id), tree.sphere_radius(node.context_id), origin, dir, min);
            return IntersectionResult {
                    t,
                    t == -1.f ? vec3(0.f) : normalize(origin + t * dir - tree.sphere_center(node.context_id)),
                    node.id
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
                return IntersectionResult { -1.f, vec3(0.f), -1 }; // Miss
            }

            if (actions.has_action(CSGActions::RetLeft) ||
                (actions.has_action(CSGActions::RetLeftIfCloser) && res_l.t <= res_r.t)) {
                return res_l;
            }

            if (actions.has_action(CSGActions::RetRight) ||
                (actions.has_action(CSGActions::RetRightIfCloser) && res_r.t <= res_l.t)) {
                if (actions.has_action(CSGActions::FlipRight)) {
                    return IntersectionResult { res_r.t, -res_r.normal, res_r.leaf_id };
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
                return IntersectionResult { -1.f, vec3(0.f), -1 }; // Miss
            }
        }
    }

    uint32_t csg_trace_ray(const csg::CSGTree& tree, const vec3& origin, const vec3& dir) {
        if (tree.get_nodes_count() <= 1) {
            return on_miss();
        }

        auto result = csg_intersect(tree, origin, dir, tree.get_node(1), 0.f);

        if (result.leaf_id == -1) {
            return on_miss();
        }

        return on_hit(origin + dir * result.t, result.normal, tree.sphere_color(tree.get_node(result.leaf_id).context_id));
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
            return on_hit(closest_hit, normalize(closest_hit - tree.sphere_center(closest_sphere)), tree.sphere_color(closest_sphere));
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
