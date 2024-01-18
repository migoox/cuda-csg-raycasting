#ifndef CUDA_CSG_RAYCASTING_CSG_TREE_H
#define CUDA_CSG_RAYCASTING_CSG_TREE_H
#include <vector>
#include <glm/glm.hpp>
#include <string>
#include <cuda_runtime.h>

namespace csg {
    struct Node {
        enum Type : int {
            None,
            Guard,
            Sphere,
            // Cube,
            UnionOp,
            DiffOp,
            InterOp
        };

        int id;      // id in the array that represents a tree
        int context_id; // represents an id of the context basing on the type, for example if type == Sphere, then
        // context_id refers to center, radius and color
        Type type;   // node type ( if None => the node is invalid)

        __host__ __device__ Node() = default;
        __host__ __device__ Node(int id, int context_id, Type type);
        __host__ __device__ int get_parent_id() const;
        __host__ __device__ int get_left_id() const;
        __host__ __device__ int get_right_id() const;
        __host__ __device__ bool is_primitive() const;
        __host__ __device__ bool is_operation() const;
    };

    class CSGTree {
    public:
        explicit CSGTree(const std::string& path);
        CSGTree();

        Node get_node(int id) const;
        size_t get_nodes_count() const { return m_node_array.size(); }
        const std::vector<csg::Node>& nodes() const { return m_node_array; }

        float sphere_radius(int context_id) const { return m_sphere_radiuses[context_id]; }
        const glm::vec3& sphere_center(int context_id) const { return m_sphere_centers[context_id]; }
        const glm::vec3& sphere_color(int context_id) const { return m_sphere_colors[context_id]; }

        const std::vector<float>& sphere_radiuses() const { return m_sphere_radiuses; }
        const std::vector<glm::vec3>& sphere_centers() const { return m_sphere_centers; }
        const std::vector<glm::vec3>& sphere_colors() const { return m_sphere_colors; }
        size_t get_sphere_count() const { return m_sphere_radiuses.size(); }

        const std::vector<float>& boundings_radiuses() const { return m_sb_radiuses; }
        const std::vector<glm::vec3>& boundings_centers() const { return m_sb_centers; }
        size_t get_operations_count() const { return m_sb_radiuses.size(); }

        void load(const std::string& path);

    private:
        static Node::Type str_to_type(const std::string& str);

        bool check_correctness(csg::Node root);

        // This function assumes that the tree is correct
        std::pair<glm::vec3, int> find_leafs_sum(csg::Node root);
        int find_furthest_leaf(csg::Node root, const glm::vec3& from);

    private:
        std::vector<Node> m_node_array;

        //.
        std::vector<float> m_sb_radiuses;
        std::vector<glm::vec3> m_sb_centers;

        // Spheres
        std::vector<float> m_sphere_radiuses;
        std::vector<glm::vec3> m_sphere_centers;
        std::vector<glm::vec3> m_sphere_colors;
    };

    enum class StackState {
        None,
        GoToLeft,
        GoToRight,
        Compute,
        LoadLeft,
        LoadRight,
        SaveLeft
    };

    enum class PointState : int {
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

        __host__ __device__ bool has_action(CSGAction action) const;

        __host__ __device__ CSGActions(PointState state_l, PointState state_r, const csg::Node& node);

        CSGAction array[3] = {None, None, None };
    };

    struct IntersectionResult {
        float t = -1.f; // if t == -1.0f => miss
        glm::vec3 normal = glm::vec3(0.f);
        int leaf_id = -1; // -1 => miss
    };
}

#endif //CUDA_CSG_RAYCASTING_CSG_TREE_H