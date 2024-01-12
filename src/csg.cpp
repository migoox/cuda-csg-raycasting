#include "csg.hpp"
#include <fstream>
#include <cmath>
#include "vendor/nlohmann/json.hpp"

csg::Node::Node(int id, int prim_id, csg::Node::Type type)
 : id(id), prim_id(prim_id), type(type) { }

int csg::Node::get_left_id() const {
    return id / 2;
}

int csg::Node::get_right_id() const {
    return 2 * id + 1;
}

int csg::Node::get_parent_id() const {
    return 2 * id;
}

csg::Node &csg::Node::operator=(const csg::Node &other) {
    if (this != &other) {
        const_cast<int&>(id) = other.id;
        const_cast<int&>(prim_id) = other.prim_id;
        const_cast<Type&>(type) = other.type;
    }
    return *this;
}

csg::Node::Node(const csg::Node &other)
: id(other.id), prim_id(other.prim_id), type(other.type) { }

csg::Node csg::CSGTree::get_node(int id) {
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
            m_radiuses.push_back(obj["radius"]);
            m_centers.emplace_back(obj["center"]["x"], obj["center"]["y"], obj["center"]["z"]);
            prim_id = m_centers.size();
        }

        m_node_array[id] = Node(id, prim_id, type);
    }

    m_node_array[0] = Node(-1, -1, Node::Guard);

    m_radiuses.shrink_to_fit();
    m_centers.shrink_to_fit();
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