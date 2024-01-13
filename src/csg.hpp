#ifndef CUDA_CSG_RAYCASTING_CSG_TREE_H
#define CUDA_CSG_RAYCASTING_CSG_TREE_H
#include <vector>
#include <glm/glm.hpp>
#include <string>

namespace csg {
    struct Node {
        enum Type {
            None,
            Guard,
            Sphere,
            // Cube,
            UnionOp,
            DiffOp,
            InterOp
        };

        const int id;      // id in the array that represents a tree
        const int prim_id; // id of a primitive ( e.g. sphere ), it's -1 if the node is not representing a primitive
        const Type type;   // node type ( if None => the node is invalid)

        Node(int id, int prim_id, Type type);

        // Copy assignment operator
        Node(const Node& other);
        Node& operator=(const Node& other);

        int get_parent_id() const;
        int get_left_id() const;
        int get_right_id() const;
    };

    class CSGTree {
    public:
        explicit CSGTree(const std::string& path);

        Node get_node(int id) const;
        size_t get_nodes_count() const { return m_node_array.size(); }

        float sphere_radius(int prim_id) const { return m_sphere_radiuses[prim_id]; }
        const glm::vec3& sphere_center(int prim_id) const { return m_sphere_centers[prim_id]; }
        const glm::vec3& sphere_color(int prim_id) const { return m_sphere_colors[prim_id]; }

        const std::vector<float>& sphere_radiuses() const { return m_sphere_radiuses; }
        const std::vector<glm::vec3>& sphere_centers() const { return m_sphere_centers; }
        const std::vector<glm::vec3>& sphere_colors() const { return m_sphere_colors; }
        size_t get_sphere_count() const { return m_sphere_radiuses.size(); }

    private:
        static Node::Type str_to_type(const std::string& str);

    private:
        std::vector<Node> m_node_array;

        // Spheres
        std::vector<float> m_sphere_radiuses;
        std::vector<glm::vec3> m_sphere_centers;
        std::vector<glm::vec3> m_sphere_colors;
    };

}

#endif //CUDA_CSG_RAYCASTING_CSG_TREE_H