#include "csg_utils.cuh"
#include <fstream>
#include <cmath>
#include <iostream>
#include "vendor/nlohmann/json.hpp"

__host__ __device__ csg::Node::Node(int id, int context_id, csg::Node::Type type)
 : id(id), context_id(context_id), type(type) { }

__host__ __device__ int csg::Node::get_left_id() const {
    if (id == 0) {
        return 1;
    }
    return 2 * id;
}

__host__ __device__ int csg::Node::get_right_id() const {
    return 2 * id + 1;
}

__host__ __device__ int csg::Node::get_parent_id() const {
    return id / 2;
}

__host__ __device__ bool csg::Node::is_primitive() const {
    if (type == csg::Node::UnionOp || type == csg::Node::DiffOp || type == csg::Node::InterOp) {
        return false;
    }
    return true;
}

__host__ __device__ bool csg::Node::is_operation() const {
    if (type == csg::Node::UnionOp || type == csg::Node::DiffOp || type == csg::Node::InterOp) {
        return true;
    }
    return false;
}

csg::Node csg::CSGTree::get_node(int id) const {
    if (id > m_node_array.size()) {
        return csg::Node(-1, -1, csg::Node::None);
    }
    return m_node_array[id];
}

bool csg::CSGTree::check_correctness(csg::Node root) {
    if (root.is_operation()) {
        if (root.get_left_id() > m_node_array.size()
            || m_node_array[root.get_left_id()].type == csg::Node::None
            || root.get_right_id() > m_node_array.size()
            || m_node_array[root.get_right_id()].type == csg::Node::None) {
            return false;
        } else {
            return check_correctness(m_node_array[root.get_left_id()]) &&
                   check_correctness(m_node_array[root.get_right_id()]);
        }
    } else {
        if ((root.get_left_id() < m_node_array.size() && m_node_array[root.get_left_id()].type != csg::Node::None)
            || (root.get_right_id() < m_node_array.size() && m_node_array[root.get_right_id()].type != csg::Node::None)) {
            return false;
        } else {
            return true;
        }
    }
}

std::pair<glm::vec3, int> csg::CSGTree::find_leafs_sum(csg::Node root) {
    if (root.is_primitive()) {
        return std::pair<glm::vec3, int>(m_sphere_centers[root.context_id], 1);
    }

    auto lft = find_leafs_sum(m_node_array[root.get_left_id()]);
    auto rght = find_leafs_sum(m_node_array[root.get_right_id()]);
    return std::pair<glm::vec3, int>(lft.first + rght.first, lft.second + rght.second);
}

int csg::CSGTree::find_furthest_leaf(csg::Node root, const glm::vec3& from) {
    if (root.is_primitive()) {
        return root.context_id;
    }

    int lft = find_furthest_leaf(m_node_array[root.get_left_id()], from);
    int rght = find_furthest_leaf(m_node_array[root.get_right_id()], from);

    if (glm::dot(m_sphere_centers[lft], from) > glm::dot(m_sphere_centers[rght], from)) {
        return lft;
    }
    return rght;
}

void csg::CSGTree::load(const std::string& path) {
    m_node_array.clear();
    m_sphere_colors.clear();
    m_sphere_radiuses.clear();
    m_sphere_centers.clear();
    m_sb_radiuses.clear();
    m_sb_centers.clear();

    using json = nlohmann::json;
    std::ifstream file(path);

    json j;
    file >> j;

    int max_id = static_cast<double>(j["scene"]["max_id"]);
    if (max_id > 1024) {
        std::cerr << "[Loader]: Unsupported tree height.\n" << std::endl;
        return;
    }

    int p = static_cast<int>(std::ceil(std::log2(max_id)));

    m_node_array.resize(std::pow(2, p), Node(-1, -1, Node::None));

    const auto& objects = j["scene"]["objects"];

    for (const auto& obj : objects) {
        int id = obj["id"];
        int context_id = -1;
        auto type = str_to_type(obj["type"]);

        if (obj["type"] == "sphere") {
            m_sphere_radiuses.push_back(obj["radius"]);
            m_sphere_centers.emplace_back(obj["center"]["x"], obj["center"]["y"], obj["center"]["z"]);
            if (obj.contains("color")) {
                m_sphere_colors.emplace_back(obj["color"]["r"], obj["color"]["g"], obj["color"]["b"]);
            } else {
                m_sphere_colors.emplace_back(1.f, 0.f, 0.f);
            }
            context_id = m_sphere_centers.size() - 1;
        }

        m_node_array[id] = Node(id, context_id, type);
    }
    m_node_array[0] = Node(0, -1, Node::Guard);

    if (!check_correctness(m_node_array[1])) {
        std::cerr << "[Loader]: The tree is incorrect -- every leaf must be a primitive and every operation must have exactly two children.\n" << std::endl;
        return;
    }

    for (int i = 1; i < m_node_array.size(); ++i) {
        if (m_node_array[i].is_operation()) {
            auto sum = find_leafs_sum(m_node_array[i]);
            glm::vec3 avg = sum.first / static_cast<float>(sum.second);
            int c_id = find_furthest_leaf(m_node_array[i], avg);

            float radius = glm::distance(m_sphere_centers[c_id], avg) + m_sphere_radiuses[c_id];
            m_sb_centers.push_back(avg);
            m_sb_radiuses.push_back(radius);
            m_node_array[i].context_id = m_sb_radiuses.size() - 1;
        }
    }

    m_node_array.shrink_to_fit();
    m_sphere_colors.shrink_to_fit();
    m_sphere_radiuses.shrink_to_fit();
    m_sphere_centers.shrink_to_fit();
    m_sb_radiuses.shrink_to_fit();
    m_sb_centers.shrink_to_fit();
}

csg::CSGTree::CSGTree() {
    m_node_array.emplace_back(0, -1, Node::Guard);
    m_node_array.emplace_back(1, 0, Node::Sphere);
    m_sphere_centers.emplace_back(0.f, 0.f, -5.f);
    m_sphere_radiuses.push_back(1.f);
    m_sphere_colors.emplace_back(1.f, 0.f, 0.f);
}

csg::CSGTree::CSGTree(const std::string &path) {
    this->load(path);
}

csg::Node::Type csg::CSGTree::str_to_type(const std::string &str) {
    if (str == "sphere") {
        return Node::Sphere;
    } else if (str == "union") {
        return Node::UnionOp;
    } else if (str == "diff") {
        return Node::DiffOp;
    } else if (str == "inter") {
        return Node::InterOp;
    }
    return Node::None;
}



__host__ __device__ csg::CSGActions::CSGActions(csg::PointState state_l, csg::PointState state_r, const csg::Node &node) {
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

__host__ __device__ bool csg::CSGActions::has_action(csg::CSGActions::CSGAction action) const {
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
