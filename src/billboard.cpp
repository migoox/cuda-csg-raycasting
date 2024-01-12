#include "billboard.hpp"

app::Billboard::Billboard()
: m_vao(0), m_vbo(0), m_ebo(0) {
    // Vertex data
    float vertices[] = {
        -1.0f, -1.0f, 0.0f, 0.f, 0.f,
         1.0f, -1.0f, 0.0f, 1.f, 0.f,
         1.0f,  1.0f, 0.0f, 1.f, 1.f,
        -1.0f,  1.0f, 0.0f, 0.f, 1.f
    };

    // Indices for two triangles
    unsigned int indices[] = {
        0, 1, 2,
        2, 3, 0,
    };

    // Vertex Array Object
    glGenVertexArrays(1, &m_vao);

    // Vertex Buffer Object
    glGenBuffers(1, &m_vbo);

    // Element Buffer Object
    glGenBuffers(1, &m_ebo);

    // Bind Vertex Array Object
    glBindVertexArray(m_vao);

    // Bind Vertex Buffer Object and copy vertex data
    glBindBuffer(GL_ARRAY_BUFFER, m_vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

    // Bind Element Buffer Object and copy index data
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_ebo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);

    // Set vertex attribute pointers
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);

    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);
}

void app::Billboard::draw() {
    // Draw the quad
    glBindVertexArray(m_vao);
    glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
}

app::Billboard::~Billboard() {
    glDeleteVertexArrays(1, &m_vao);
    glDeleteBuffers(1, &m_vbo);
    glDeleteBuffers(1, &m_ebo);
}
