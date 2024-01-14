#include "csg_utils.cuh"
#include <fstream>
#include <cmath>
#include "vendor/nlohmann/json.hpp"

__host__ __device__ csg::Node::Node(int id, int prim_id, csg::Node::Type type)
 : id(id), prim_id(prim_id), type(type) { }

__host__ __device__ int csg::Node::get_left_id() const {
    return 2 * id;
}

__host__ __device__ int csg::Node::get_right_id() const {
    return 2 * id + 1;
}

__host__ __device__ int csg::Node::get_parent_id() const {
    return id / 2;
}

csg::Node csg::CSGTree::get_node(int id) const {
    if (id > m_node_array.size()) {
        return csg::Node(-1, -1, csg::Node::None);
    }
    return m_node_array[id];
}

csg::CSGTree::CSGTree(const std::string &path) {
    using json = nlohmann::json;
    std::ifstream file(path);

    json j;
    file >> j;

    int p = static_cast<int>(std::ceil(std::log2(static_cast<double>(j["scene"]["max_id"]))));
    m_node_array.resize(std::pow(2, p), Node(-1, -1, Node::None));

    const auto& objects = j["scene"]["objects"];

    for (const auto& obj : objects) {
        int id = obj["id"];
        int prim_id = -1;
        auto type = str_to_type(obj["type"]);

        if (obj["type"] == "sphere") {
            m_sphere_radiuses.push_back(obj["radius"]);
            m_sphere_centers.emplace_back(obj["center"]["x"], obj["center"]["y"], obj["center"]["z"]);
            if (obj.contains("color")) {
                m_sphere_colors.emplace_back(obj["color"]["r"], obj["color"]["g"], obj["color"]["b"]);
            } else {
                m_sphere_colors.emplace_back(1.f, 0.f, 0.f);
            }
            prim_id = m_sphere_centers.size() - 1;
        }

        m_node_array[id] = Node(id, prim_id, type);
    }

    m_node_array[0] = Node(-1, -1, Node::Guard);

    m_sphere_radiuses.shrink_to_fit();
    m_sphere_colors.shrink_to_fit();
    m_sphere_centers.shrink_to_fit();
    m_node_array.shrink_to_fit();
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
