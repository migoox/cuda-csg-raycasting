#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include "cuda_raycaster.cuh"

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
__device__ float get_sphere_hit(glm::vec3 center, float radius, glm::vec3 ray_origin, glm::vec3 ray_dir, float min) {
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

__device__ uint32_t on_hit(glm::vec3 hit_point, glm::vec3 normal, glm::vec3 color) {
    glm::vec3 light_pos = glm::vec3(0.f, 1.f, 0.f);

    // normal = 0.5f * (normal + 1.f);
    // return get_color_rgb_norm(normal.r, normal.g, normal.b);

    glm::vec3 light_dir = normalize(light_pos - hit_point);

    glm::vec3 res_color = color * glm::clamp(glm::dot(normal, light_dir), 0.f, 1.f);

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
__global__ void init(uint32_t *canvas, glm::vec3 *origins, glm::vec3 *dirs, int count) {
    uint32_t k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k < count) {
        canvas[k] = 0x000000FF;
        origins[k] = glm::vec3(0.f);
        dirs[k] = glm::vec3(0.f);
    }
}

cuda_raycaster::GPURayCaster::GPURayCaster(const csg::CSGTree& tree, int width, int height)
: m_width(width), m_height(height) {
    cudaError_t cuda_status;

    // Allocate memory on the device using cudaMalloc
    cuda_status = cudaMalloc((void**)&m_dev_canvas, width * height * sizeof(uint32_t));
    check_cuda_error(cuda_status, "[CUDA]: cudaMalloc failed: ");
    cuda_status = cudaMalloc((void**)&m_dev_origins, width * height * sizeof(glm::vec3));
    check_cuda_error(cuda_status, "[CUDA]: cudaMalloc failed: ");
    cuda_status = cudaMalloc((void**)&m_dev_dirs, width * height * sizeof(glm::vec3));
    check_cuda_error(cuda_status, "[CUDA]: cudaMalloc failed: ");

    size_t threads_per_block = 1024;
    size_t blocks_num = m_width * m_height / threads_per_block + 1;
    init<<<blocks_num, threads_per_block>>>(
            m_dev_canvas,
            m_dev_origins,
            m_dev_dirs,
            width * height
    );
    cudaDeviceSynchronize();

    cuda_status = cudaMalloc((void**)&m_dev_radiuses, tree.get_sphere_count() * sizeof(float));
    check_cuda_error(cuda_status, "[CUDA]: cudaMalloc failed: ");
    cuda_status = cudaMalloc((void**)&m_dev_centers, tree.get_sphere_count() * sizeof(glm::vec3));
    check_cuda_error(cuda_status, "[CUDA]: cudaMalloc failed: ");
    cuda_status = cudaMalloc((void**)&m_dev_colors, tree.get_sphere_count() * sizeof(glm::vec3));
    check_cuda_error(cuda_status, "[CUDA]: cudaMalloc failed: ");
    cuda_status = cudaMalloc((void**)&m_dev_node_array, tree.get_nodes_count() * sizeof(csg::Node));
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

    m_spheres_count = tree.get_sphere_count();
    m_nodes_count = tree.get_nodes_count();
}

cuda_raycaster::GPURayCaster::~GPURayCaster() {
    cudaFree(m_dev_origins);
    cudaFree(m_dev_dirs);
    cudaFree(m_dev_canvas);

    cudaFree(m_dev_radiuses);
    cudaFree(m_dev_centers);
    cudaFree(m_dev_colors);
    cudaFree(m_dev_node_array);
}

__global__ void find_dirs(
        glm::vec2 canvas_size,
        glm::vec3 eye,
        glm::mat4 inv_proj,
        glm::mat4 inv_view,
        glm::vec3* origins,
        glm::vec3* dirs,
        int width,
        int height
) {
    uint32_t k = blockIdx.x * blockDim.x + threadIdx.x;
    int count = width * height;
    if (k < count) {
        uint32_t x = k % width;
        uint32_t y = k / width;
        glm::vec2 viewport_coords = { static_cast<float>(x) / canvas_size.x, (static_cast<float>(y)) / canvas_size.y };
        viewport_coords = viewport_coords * 2.0f - 1.0f;

        // pixel's position in the world space
        glm::vec4 target = inv_proj * glm::vec4(viewport_coords.x, viewport_coords.y, -1.f, 1.f);
        dirs[k] = glm::vec3(inv_view * glm::vec4(normalize(glm::vec3(target) / target.w), 0.f)); // world space
        origins[k] = eye;
    }
}

__global__ void trace_ray(
        float *radiuses,
        glm::vec3 *centers,
        glm::vec3 *colors,
        uint32_t spheres_count,
        glm::vec3 *origins,
        glm::vec3 *dirs,
        uint32_t *canvas,
        int width,
        int height
) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= width * height) {
        return;
    }

    float t_min = FLT_MAX;
    int closest_sphere = -1;
    for (int i = 0; i < spheres_count; ++i) {
        float t = get_sphere_hit(centers[i], radiuses[i], origins[k], dirs[k], 0.f);

        if (t > 0.f && t < t_min) {
            t_min = t;
            closest_sphere = i;
        }
    }

    if (closest_sphere != -1) {
        glm::vec3 closest_hit = origins[k] + dirs[k] * t_min;
        canvas[k] = on_hit(closest_hit, normalize(closest_hit - centers[closest_sphere]),
                           colors[closest_sphere]);
    } else {
        canvas[k] = on_miss();
    }
}

void cuda_raycaster::GPURayCaster::update_canvas(renderer::Image &canvas,
                                                 const cuda_raycaster::GPURayCaster::Input &input) {
    resize(canvas.get_width(), canvas.get_height());
    size_t threads_per_block = 1024;
    size_t blocks_num = m_width * m_height / threads_per_block + 1;

    // First kernel: find rays and origins
    find_dirs<<<blocks_num, threads_per_block>>>(
            input.canvas,
            input.eye,
            input.inv_proj,
            input.inv_view,
            m_dev_origins,
            m_dev_dirs,
            m_width,
            m_height
    );
    cudaDeviceSynchronize();

    // Second kernel: ray casting
    trace_ray<<<blocks_num, threads_per_block>>>(
            m_dev_radiuses,
            m_dev_centers,
            m_dev_colors,
            m_spheres_count,
            m_dev_origins,
            m_dev_dirs,
            m_dev_canvas,
            m_width,
            m_height
    );
    cudaDeviceSynchronize();

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

    cudaFree(m_dev_dirs);
    cudaFree(m_dev_origins);
    cudaFree(m_dev_canvas);

    cudaError_t cuda_status;
    cuda_status = cudaMalloc((void**)&m_dev_origins, width * height * sizeof(glm::vec3));
    check_cuda_error(cuda_status, "[CUDA]: cudaMalloc failed: ");
    cuda_status = cudaMalloc((void**)&m_dev_dirs, width * height * sizeof(glm::vec3));
    check_cuda_error(cuda_status, "[CUDA]: cudaMalloc failed: ");
    cuda_status = cudaMalloc((void**)&m_dev_canvas, width * height * sizeof(uint32_t));
    check_cuda_error(cuda_status, "[CUDA]: cudaMalloc failed: ");
}

