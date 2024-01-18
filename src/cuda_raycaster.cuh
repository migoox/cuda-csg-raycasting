#ifndef CUDA_CSG_RAYCASTING_CUDA_RAYCASTER_CUH
#define CUDA_CSG_RAYCASTING_CUDA_RAYCASTER_CUH
#include "csg_utils.cuh"
#include "textures.hpp"

namespace cuda_raycaster {
    class GPURayCaster {
    public:
        struct Input {
            glm::vec3 sun;
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

        void set_tree(const csg::CSGTree& tree);
        void resize(int width, int height);

        void update_canvas(renderer::Image& canvas, const Input& input);

    private:
        void load_tree(const csg::CSGTree &tree);


    private:
        int m_width, m_height;
        uint32_t *m_dev_canvas{};

        float* m_dev_boundings_radiuses{};
        glm::vec3* m_dev_boundings_centers{};
        uint32_t m_operations_count;

        float *m_dev_radiuses{};
        glm::vec3 *m_dev_centers{};
        glm::vec3 *m_dev_colors{};
        uint32_t m_spheres_count;

        csg::Node *m_dev_node_array{};
        uint32_t m_nodes_count;
    };
}

#endif //CUDA_CSG_RAYCASTING_CUDA_RAYCASTER_CUH
