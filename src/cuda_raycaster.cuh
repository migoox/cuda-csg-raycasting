#ifndef CUDA_CSG_RAYCASTING_CUDA_RAYCASTER_CUH
#define CUDA_CSG_RAYCASTING_CUDA_RAYCASTER_CUH
#include "csg_utils.cuh"
#include "textures.hpp"

namespace cuda_raycaster {
    class GPURayCaster {
    public:
        struct Input {
            glm::mat4 inv_proj;
            glm::mat4 inv_view;
            glm::vec3 eye;
            glm::vec2 canvas;
            const csg::CSGTree& tree;
            bool show_csg;
        };

        GPURayCaster() = delete;
        ~GPURayCaster();
        GPURayCaster(const csg::CSGTree& tree, int width, int height);

        void resize(int width, int height);

        void update_canvas(renderer::Image& canvas, const Input& input);

    private:
        int m_width, m_height;
        glm::vec3 *m_dev_origins{};
        glm::vec3 *m_dev_dirs{};
        uint32_t *m_dev_canvas{};

        float *m_dev_radiuses{};
        glm::vec3 *m_dev_centers{};
        glm::vec3 *m_dev_colors{};
        uint32_t m_spheres_count;

        csg::Node *m_dev_node_array{};
        uint32_t m_nodes_count;
    };

}

#endif //CUDA_CSG_RAYCASTING_CUDA_RAYCASTER_CUH
