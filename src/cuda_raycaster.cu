#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include "cuda_raycaster.cuh"
#include <cuda_runtime.h>
#include "cuda_stack.cuh"
__device__ glm::vec3 sun_pos;
__device__ glm::vec3 cam_pos;

__device__ uint32_t get_color_rgb_norm(float r, float g, float b) {
    return static_cast<uint32_t>(std::round(r * 255.f)) << 24 |
           static_cast<uint32_t>(std::round(g * 255.f)) << 16 |
           static_cast<uint32_t>(std::round(b * 255.f)) << 8 |
           0xFF;
}

__device__ uint32_t get_color_rgb(uint8_t r, uint8_t g, uint8_t b) {
    return static_cast<uint32_t>(r) << 24 |
           static_cast<uint32_t>(g) << 16 |
           static_cast<uint32_t>(b) << 8 |
           0xFF;
}

// Returns parametric function argument (t), returns -1.f in the case there was no hit
__device__ float get_sphere_hit(const glm::vec3& center, float radius, const glm::vec3& ray_origin, const glm::vec3& ray_dir, float min) {
    glm::vec3 co = ray_origin - center;

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

__device__ uint32_t on_hit(const glm::vec3& hit_point, const glm::vec3& normal, const glm::vec3& color) {
    // normal = 0.5f * (normal + 1.f);
    // return get_color_rgb_norm(normal.r, normal.g, normal.b);

    // Assuming that sun_pos is normalized
    glm::vec3 light_dir = sun_pos;
    // Ambient + Diffuse + Specular
    glm::vec3 res_color = color * glm::clamp(
            0.1f +
            glm::clamp(glm::dot(normal, light_dir), 0.f, 1.f) +
            pow(glm::clamp(glm::dot(
                    glm::normalize(cam_pos - hit_point),
                    glm::reflect(-light_dir, normal)), 0.f, 1.f), 32.f), 0.f, 1.f
    );

    return get_color_rgb_norm(res_color.r, res_color.g, res_color.b);
}

__device__ uint32_t on_miss() {
    // Background color
    return get_color_rgb(60, 60, 60);
}

void check_cuda_error(const cudaError_t &cuda_status, const char *msg) {
    if (cuda_status != cudaSuccess) {
        std::cerr << msg << cudaGetErrorString(cuda_status) << std::endl;
        std::terminate();
    }
}


void cuda_raycaster::GPURayCaster::load_tree(const csg::CSGTree &tree) {
    cudaError_t cuda_status;
    cuda_status = cudaMalloc((void**)&m_dev_radiuses, tree.get_sphere_count() * sizeof(float));
    check_cuda_error(cuda_status, "[CUDA]: cudaMalloc failed: ");
    cuda_status = cudaMalloc((void**)&m_dev_centers, tree.get_sphere_count() * sizeof(glm::vec3));
    check_cuda_error(cuda_status, "[CUDA]: cudaMalloc failed: ");
    cuda_status = cudaMalloc((void**)&m_dev_colors, tree.get_sphere_count() * sizeof(glm::vec3));
    check_cuda_error(cuda_status, "[CUDA]: cudaMalloc failed: ");
    cuda_status = cudaMalloc((void**)&m_dev_node_array, tree.get_nodes_count() * sizeof(csg::Node));
    check_cuda_error(cuda_status, "[CUDA]: cudaMalloc failed: ");
    cuda_status = cudaMalloc((void**)&m_dev_boundings_radiuses, tree.get_operations_count() * sizeof(float));
    check_cuda_error(cuda_status, "[CUDA]: cudaMalloc failed: ");
    cuda_status = cudaMalloc((void**)&m_dev_boundings_centers, tree.get_operations_count() * sizeof(glm::vec3));
    check_cuda_error(cuda_status, "[CUDA]: cudaMalloc failed: ");

    // Move tree data to the gpu
    cuda_status = cudaMemcpy(m_dev_radiuses, tree.sphere_radiuses().data(), tree.get_sphere_count() * sizeof(float), cudaMemcpyHostToDevice);
    check_cuda_error(cuda_status, "[CUDA]: cudaMemcpy failed: ");
    cuda_status = cudaMemcpy(m_dev_centers, tree.sphere_centers().data(), tree.get_sphere_count() * sizeof(glm::vec3), cudaMemcpyHostToDevice);
    check_cuda_error(cuda_status, "[CUDA]: cudaMemcpy failed: ");
    cuda_status = cudaMemcpy(m_dev_colors, tree.sphere_colors().data(), tree.get_sphere_count() * sizeof(glm::vec3), cudaMemcpyHostToDevice);
    check_cuda_error(cuda_status, "[CUDA]: cudaMemcpy failed: ");
    cuda_status = cudaMemcpy(m_dev_node_array, tree.nodes().data(), tree.get_nodes_count() * sizeof(csg::Node), cudaMemcpyHostToDevice);
    check_cuda_error(cuda_status, "[CUDA]: cudaMemcpy failed: ");
    cuda_status = cudaMemcpy(m_dev_boundings_radiuses, tree.boundings_radiuses().data(), tree.get_operations_count() * sizeof(float), cudaMemcpyHostToDevice);
    check_cuda_error(cuda_status, "[CUDA]: cudaMemcpy failed: ");
    cuda_status = cudaMemcpy(m_dev_boundings_centers, tree.boundings_centers().data(), tree.get_operations_count() * sizeof(glm::vec3), cudaMemcpyHostToDevice);
    check_cuda_error(cuda_status, "[CUDA]: cudaMemcpy failed: ");

    m_spheres_count = tree.get_sphere_count();
    m_nodes_count = tree.get_nodes_count();
    m_operations_count = tree.get_operations_count();
}

void cuda_raycaster::GPURayCaster::set_tree(const csg::CSGTree &tree) {
    cudaFree(m_dev_radiuses);
    cudaFree(m_dev_centers);
    cudaFree(m_dev_colors);
    cudaFree(m_dev_node_array);
    cudaFree(m_dev_boundings_radiuses);
    cudaFree(m_dev_boundings_centers);

    load_tree(tree);
}

cuda_raycaster::GPURayCaster::GPURayCaster(const csg::CSGTree& tree, int width, int height)
: m_width(width), m_height(height) {
    cudaError_t cuda_status;

    // Allocate memory on the device using cudaMalloc
    cuda_status = cudaMalloc((void**)&m_dev_canvas, width * height * sizeof(uint32_t));
    check_cuda_error(cuda_status, "[CUDA]: cudaMalloc failed: ");

    cudaDeviceSynchronize();
    load_tree(tree);
}


cuda_raycaster::GPURayCaster::~GPURayCaster() {
    cudaFree(m_dev_canvas);
    cudaFree(m_dev_radiuses);
    cudaFree(m_dev_centers);
    cudaFree(m_dev_colors);
    cudaFree(m_dev_node_array);
    cudaFree(m_dev_boundings_radiuses);
    cudaFree(m_dev_boundings_centers);
}

__device__ csg::PointState csg_point_classify(const csg::IntersectionResult& res, glm::vec3 ray_dir) {
    if (res.leaf_id == -1) {
        return csg::PointState::Miss;
    }

    if (dot(res.normal, ray_dir) > 0.f) {
        return csg::PointState::Exit;
    }

    if (dot(res.normal, ray_dir) < 0.f) {
        return csg::PointState::Enter;
    }

    return csg::PointState::Miss;
}

__device__ csg::IntersectionResult csg_intersect_stack(
        csg::Node *nodes,
        int prim_count,
        int nodes_count,
        float *radiuses,
        glm::vec3 *centers,
        float *boundings_radiuses,
        glm::vec3 *boundings_centers,
        glm::vec3 origin,
        glm::vec3 dir
) {
    cuda_utils::Stack<csg::IntersectionResult, 64> prim_stack;
    cuda_utils::Stack<csg::StackState, 64> states_stack;
    cuda_utils::Stack<float, 64> min_stack;

    float curr_min = 0;
    csg::Node curr_node = nodes[0]; // guard
    csg::StackState curr_state = csg::StackState::GoToLeft;

    auto res_l = csg::IntersectionResult {
            -1.f,
            glm::vec3(0.f),
            -1
    }; // invalid result
    auto res_r = csg::IntersectionResult {
            -1.f,
            glm::vec3(0.f),
            -1
    }; // invalid result

    while (true) {
        if (curr_state == csg::StackState::SaveLeft) {
            curr_min = min_stack.pop();
            prim_stack.push(res_l);
            curr_state = csg::StackState::GoToRight;
        }

        if (curr_state == csg::StackState::GoToLeft || curr_state == csg::StackState::GoToRight) {
            // Go to the desired node
            if (curr_state == csg::StackState::GoToLeft) {
                curr_node = nodes[curr_node.get_left_id()];
            } else {
                curr_node = nodes[curr_node.get_right_id()];
            }

            // Update stacks
            if (curr_node.is_operation()) {

                // There are 4 possible cases:
                //      1. both children are Primitives -- goto_L == false, goto_R == false,
                //      2. left child is a Primitive -- goto_L == false, goto_R == true,
                //      3. right child is a Primitive -- goto_L == true, goto_R == false,
                //      4. both children are Operations -- goto_L == true, goto_R == true.
                //
                //      If ray is out of the sphere bounding it's equivalent of Primitive Miss.
                //

                bool goto_l = true;
                bool goto_r = true;

                if (nodes[curr_node.get_left_id()].is_primitive()) {
                    goto_l = false;
                    float t = get_sphere_hit(
                            centers[nodes[curr_node.get_left_id()].context_id],
                            radiuses[nodes[curr_node.get_left_id()].context_id],
                            origin,
                            dir,
                            curr_min
                    );
                    res_l = csg::IntersectionResult {
                            t,
                            t < 0.f ? glm::vec3(0.f) : normalize(origin + t * dir - centers[nodes[curr_node.get_left_id()].context_id]),
                            t < 0.f ? -1 : curr_node.get_left_id()
                    };
                } else {
                    // Check sphere bounding (not working yet)
//                    float t = get_sphere_hit(
//                            boundings_centers[nodes[curr_node.get_left_id()].context_id],
//                            boundings_radiuses[nodes[curr_node.get_left_id()].context_id],
//                            origin,
//                            dir,
//                            curr_min
//                    );
////
//                    if (t < 0.f) {
//                        goto_l = false;
//                        res_l = csg::IntersectionResult {
//                            -1.f,
//                            glm::vec3(0.f),
//                            -1
//                        };
//                    }
                }

                if (nodes[curr_node.get_right_id()].is_primitive()) {
                    goto_r = false;
                    float t = get_sphere_hit(
                            centers[nodes[curr_node.get_right_id()].context_id],
                            radiuses[nodes[curr_node.get_right_id()].context_id],
                            origin,
                            dir,
                            curr_min
                    );
                    res_r = csg::IntersectionResult {
                            t,
                            t < 0.f ? glm::vec3(0.f) : normalize(origin + t * dir - centers[nodes[curr_node.get_right_id()].context_id]),
                            t < 0.f ? -1 : curr_node.get_right_id()
                    };
                } else {
                    // Check sphere bounding (not working yet)
//                    float t = get_sphere_hit(
//                            boundings_centers[nodes[curr_node.get_right_id()].context_id],
//                            boundings_radiuses[nodes[curr_node.get_right_id()].context_id],
//                            origin,
//                            dir,
//                            curr_min
//                    );
////
//                    if (t < 0.f) {
//                        goto_r = false;
//                        res_r = csg::IntersectionResult {
//                            -1.f,
//                            glm::vec3(0.f),
//                            -1
//                        };
//                    }
                }

                if (!goto_l && !goto_r) { // case 1
                    // both (t_L, N_L) and (t_R, N_R) are calculated
                    curr_state = csg::StackState::Compute;
                } else if (!goto_l && goto_r) { // case 2
                    prim_stack.push(res_l);
                    states_stack.push(csg::StackState::LoadLeft);
                    curr_state = csg::StackState::GoToRight;
                } else if (goto_l && !goto_r) { // case 3
                    prim_stack.push(res_r);
                    states_stack.push(csg::StackState::LoadRight);
                    curr_state = csg::StackState::GoToLeft;
                } else { // case 4
                    min_stack.push(curr_min);
                    states_stack.push(csg::StackState::LoadLeft);
                    states_stack.push(csg::StackState::SaveLeft);
                    curr_state = csg::StackState::GoToLeft; // w assume that left child has a priority
                }

            } else {
                // curr_node is a Primitive. This section will be invoked if LoopRight or LoopLeft
                // has been called, or if the root is a primitive
                float t = get_sphere_hit(
                        centers[curr_node.context_id],
                        radiuses[curr_node.context_id],
                        origin,
                        dir,
                        curr_min
                );
                if (curr_state == csg::StackState::GoToLeft) {
                    res_l = csg::IntersectionResult {
                            t,
                            t < 0.f ? glm::vec3(0.f) : normalize(origin + t * dir - centers[curr_node.context_id]),
                            t < 0.f ? -1 : curr_node.id
                    };
                } else { // GoToRight
                    res_r = csg::IntersectionResult {
                            t,
                            t < 0.f ? glm::vec3(0.f) : normalize(origin + t * dir - centers[curr_node.context_id]),
                            t < 0.f ? -1 : curr_node.id
                    };
                }
                if (states_stack.is_empty()) {
                    break;
                }
                curr_state = states_stack.pop();
                curr_node = nodes[curr_node.get_parent_id()];
            }
        }

        if (curr_state == csg::StackState::LoadLeft) {
            res_l = prim_stack.pop();
            curr_state = csg::StackState::Compute;
        }

        if (curr_state == csg::StackState::LoadRight) {
            res_r = prim_stack.pop();
            curr_state = csg::StackState::Compute;
        }

        if (curr_state == csg::StackState::Compute) {
            //printf("compute \n");

            csg::PointState state_l = csg_point_classify(res_l, dir);
            csg::PointState state_r = csg_point_classify(res_r, dir);

            csg::CSGActions actions = csg::CSGActions(state_l, state_r, curr_node);

            if (actions.has_action(csg::CSGActions::RetLeft) ||
                (actions.has_action(csg::CSGActions::RetLeftIfCloser) && res_l.t <= res_r.t)) {
                res_r = res_l;

                if (states_stack.is_empty()) {
                    break;
                }

                curr_state = states_stack.pop();
                curr_node = nodes[curr_node.get_parent_id()];

            } else if (actions.has_action(csg::CSGActions::RetRight) ||
                       (actions.has_action(csg::CSGActions::RetRightIfCloser) && res_r.t <= res_l.t))  {

                if (actions.has_action(csg::CSGActions::FlipRight)) {
                    res_r.normal = -res_r.normal;
                }
                res_l = res_r;

                if (states_stack.is_empty()) {
                    break;
                }

                curr_state = states_stack.pop();
                curr_node = nodes[curr_node.get_parent_id()];

            } else if (actions.has_action(csg::CSGActions::LoopLeft) ||
                       (actions.has_action((csg::CSGActions::LoopLeftIfCloser)) && res_l.t <= res_r.t)) {
                curr_min = res_l.t;
                prim_stack.push(res_r);
                states_stack.push(csg::StackState::LoadRight);
                curr_state = csg::StackState::GoToLeft;
            } else if (actions.has_action(csg::CSGActions::LoopRight) ||
                       (actions.has_action((csg::CSGActions::LoopRightIfCloser)) && res_r.t <= res_l.t)) {
                curr_min = res_r.t;
                prim_stack.push(res_l);
                states_stack.push(csg::StackState::LoadLeft);
                curr_state = csg::StackState::GoToRight;
            } else { // Miss
                res_l = csg::IntersectionResult { -1.f, glm::vec3(0.f), -1 }; // invalid
                res_r = csg::IntersectionResult { -1.f, glm::vec3(0.f), -1 }; // invalid

                if (states_stack.is_empty()) {
                    break;
                }

                curr_state = states_stack.pop();
                curr_node = nodes[curr_node.get_parent_id()];
            }
        }

        if (curr_node.id == 0) {
            break;
        }
    }

    return res_l;
}

__global__ void csg_trace_ray_stack(
         csg::Node *nodes,
        float *radiuses,
        glm::vec3 *centers,
        glm::vec3 *colors,
        float *boundings_radiuses,
        glm::vec3 *boundings_centers,
        int prim_count,
        int nodes_count,
        uint32_t *canvas,
        int width,
        int height,
        glm::vec3 eye,
        glm::vec2 canvas_size,
        glm::mat4 inv_proj,
        glm::mat4 inv_view
) {
uint32_t k = blockIdx.x * blockDim.x + threadIdx.x;
    int count = width * height;
    if (k >= count) {
        return;
    }

    uint32_t x = k % width;
    uint32_t y = k / width;
    glm::vec2 viewport_coords = { static_cast<float>(x) / canvas_size.x, (static_cast<float>(y)) / canvas_size.y };
    viewport_coords = viewport_coords * 2.0f - 1.0f;

    // pixel's position in the world space
    glm::vec4 target = inv_proj * glm::vec4(viewport_coords.x, viewport_coords.y, -1.f, 1.f);
    glm::vec3 dir = glm::vec3(inv_view * glm::vec4(normalize(glm::vec3(target) / target.w), 0.f)); // world space
    glm::vec3 origin = eye;

    // Assuming that there is max 512 leafs and max 512 centers
    __shared__ float sm_radiuses[512];
    __shared__ glm::vec3 sm_centers[512];
    __shared__ csg::Node sm_nodes[512];

    // Assuming that there are exactly 1024 threads in one block
    if (threadIdx.x < prim_count) {
        sm_radiuses[threadIdx.x] = radiuses[threadIdx.x];
        sm_centers[threadIdx.x] = centers[threadIdx.x];
    }

    if (threadIdx.x < nodes_count) {
        sm_nodes[threadIdx.x] = nodes[threadIdx.x];
    }

    // Sync the threads within the block
    __syncthreads();

    if (k >= count) {
        return;
    }

    if (nodes_count <= 1) {
        canvas[k] = on_miss();
    }

    auto result = csg_intersect_stack(sm_nodes, prim_count, nodes_count, sm_radiuses, sm_centers, boundings_radiuses, boundings_centers, origin, dir);

    if (result.leaf_id == -1) {
        canvas[k] = on_miss();
    } else {
        canvas[k] = on_hit(origin + dir * result.t, result.normal, colors[sm_nodes[result.leaf_id].context_id]);
    }
}

__global__ void trace_ray(
        float *radiuses,
        glm::vec3 *centers,
        glm::vec3 *colors,
        uint32_t spheres_count,
        uint32_t *canvas,
        int width,
        int height,
        glm::vec3 eye,
        glm::vec2 canvas_size,
        glm::mat4 inv_proj,
        glm::mat4 inv_view
) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= width * height) {
        return;
    }

    uint32_t x = k % width;
    uint32_t y = k / width;
    glm::vec2 viewport_coords = { static_cast<float>(x) / canvas_size.x, (static_cast<float>(y)) / canvas_size.y };
    viewport_coords = viewport_coords * 2.0f - 1.0f;

    // pixel's position in the world space
    glm::vec4 target = inv_proj * glm::vec4(viewport_coords.x, viewport_coords.y, -1.f, 1.f);
    glm::vec3 dir = glm::vec3(inv_view * glm::vec4(normalize(glm::vec3(target) / target.w), 0.f)); // world space
    glm::vec3 origin = eye;

    float t_min = FLT_MAX;
    int closest_sphere = -1;
    for (int i = 0; i < spheres_count; ++i) {
        float t = get_sphere_hit(centers[i], radiuses[i], origin, dir, 0.f);

        if (t > 0.f && t < t_min) {
            t_min = t;
            closest_sphere = i;
        }
    }

    if (closest_sphere != -1) {
        glm::vec3 closest_hit = origin + dir * t_min;
        canvas[k] = on_hit(closest_hit, normalize(closest_hit - centers[closest_sphere]),
                           colors[closest_sphere]);
    } else {
        canvas[k] = on_miss();
    }
}

void cuda_raycaster::GPURayCaster::update_canvas(renderer::Image &canvas,
                                                 const cuda_raycaster::GPURayCaster::Input &input) {
    glm::vec3 sun = glm::normalize(input.sun);
    cudaMemcpyToSymbol(sun_pos, &sun, sizeof(glm::vec3));
    cudaMemcpyToSymbol(cam_pos, &input.eye, sizeof(glm::vec3));
    resize(canvas.get_width(), canvas.get_height());
    size_t threads_per_block = 1024;
    size_t blocks_num = m_width * m_height / threads_per_block + 1;

    if (input.show_csg) {
        csg_trace_ray_stack<<<blocks_num, threads_per_block>>>(
                m_dev_node_array,
                m_dev_radiuses,
                m_dev_centers,
                m_dev_colors,
                m_dev_boundings_radiuses,
                m_dev_boundings_centers,
                m_spheres_count,
                m_nodes_count,
                m_dev_canvas,
                m_width,
                m_height,
                input.eye,
                input.canvas,
                input.inv_proj,
                input.inv_view
        );
    } else {
        trace_ray<<<blocks_num, threads_per_block>>>(
                m_dev_radiuses,
                m_dev_centers,
                m_dev_colors,
                m_spheres_count,
                m_dev_canvas,
                m_width,
                m_height,
                input.eye,
                input.canvas,
                input.inv_proj,
                input.inv_view
        );
    }
    cudaDeviceSynchronize();

    // Copy the canvas data to the cpu
    cudaError_t cuda_status;
    cuda_status = cudaMemcpy((void*)canvas.raw(), m_dev_canvas, m_width * m_height * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    check_cuda_error(cuda_status, "[CUDA]: cudaMemcpy failed: ");
}

void cuda_raycaster::GPURayCaster::resize(int width, int height) {
    if (width == m_width && height == m_height) {
        return;
    }
    m_width = width;
    m_height = height;

    cudaFree(m_dev_canvas);

    cudaError_t cuda_status;
    cuda_status = cudaMalloc((void**)&m_dev_canvas, width * height * sizeof(uint32_t));
    check_cuda_error(cuda_status, "[CUDA]: cudaMalloc failed: ");
}


