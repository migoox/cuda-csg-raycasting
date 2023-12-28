#include "image.hpp"
#include <cmath>


img::Image::Image(uint32_t width, uint32_t height) : m_width(width), m_height(height) {
    m_data = std::vector<uint32_t>(m_width * m_height, 0x00000000);
}

void img::Image::clear(uint32_t color) {
    for (auto& pixel : m_data) {
        pixel = color;
    }
}

void img::Image::set_pixel_safe(uint32_t x, uint32_t y, uint32_t color) {
    if (!in_bounds(x, y)) {
        return;
    }

    m_data[x + y * m_width] = color;
}

void img::Image::set_pixel(uint32_t x, uint32_t y, uint32_t color) {
    m_data[x + y * m_width] = color;
}

void img::Image::resize(uint32_t new_width, uint32_t new_height) {
    m_data.resize(new_width * new_height, 0x00000000);
    m_width = new_width;
    m_height = new_height;
}

bool img::Image::in_bounds(uint32_t x, uint32_t y) const {
    return x < m_width && y < m_height;
}

uint32_t img::Image::get_pixel(uint32_t x, uint32_t y) const {
    return m_data[x + y * m_width];
}

uint32_t img::get_color_rgb_norm(float r, float g, float b) {
    return static_cast<uint32_t>(std::round(r * 255.f)) << 24 |
           static_cast<uint32_t>(std::round(g * 255.f)) << 16 |
           static_cast<uint32_t>(std::round(b * 255.f)) << 8 |
           0xFF;
}

uint32_t img::get_color_rgb(uint8_t r, uint8_t g, uint8_t b) {
    return static_cast<uint32_t>(r) << 24 |
           static_cast<uint32_t>(g) << 16 |
           static_cast<uint32_t>(b) << 8 |
           0xFF;
}

uint32_t img::get_color_rgba_norm(float r, float g, float b, float a) {
    return static_cast<uint32_t>(std::round(r * 255.f)) << 24 |
           static_cast<uint32_t>(std::round(g * 255.f)) << 16 |
           static_cast<uint32_t>(std::round(b * 255.f)) << 8 |
           static_cast<uint32_t>(std::round(a * 255.f));
}

uint32_t img::get_color_rgba(uint8_t r, uint8_t g, uint8_t b, uint8_t a) {
    return static_cast<uint32_t>(r) << 24 |
           static_cast<uint32_t>(g) << 16 |
           static_cast<uint32_t>(b) << 8 |
           static_cast<uint32_t>(a);
}

float img::get_red_norm(uint32_t color) {
    // Extract the red component from the uint32_t color and normalize it to the range [0, 1]
    return static_cast<float>((color >> 24) & 0xFF) / 255.f;
}

uint8_t img::get_red(uint32_t color) {
    // Extract and return the red component as an 8-bit unsigned integer
    return static_cast<uint8_t>((color >> 24) & 0xFF);
}

float img::get_green_norm(uint32_t color) {
    // Extract the green component from the uint32_t color and normalize it to the range [0, 1]
    return static_cast<float>((color >> 16) & 0xFF) / 255.f;
}

uint8_t img::get_green(uint32_t color) {
    // Extract and return the green component as an 8-bit unsigned integer
    return static_cast<uint8_t>((color >> 16) & 0xFF);
}

float img::get_blue_norm(uint32_t color) {
    // Extract the blue component from the uint32_t color and normalize it to the range [0, 1]
    return static_cast<float>((color >> 8) & 0xFF) / 255.f;
}

uint8_t img::get_blue(uint32_t color) {
    // Extract and return the blue component as an 8-bit unsigned integer
    return static_cast<uint8_t>((color >> 8) & 0xFF);
}

float img::get_alpha_norm(uint32_t color) {
    // Extract the alpha component from the uint32_t color and normalize it to the range [0, 1]
    return static_cast<float>(color & 0xFF) / 255.f;
}

uint8_t img::get_alpha(uint32_t color) {
    // Extract and return the alpha component as an 8-bit unsigned integer
    return static_cast<uint8_t>(color & 0xFF);
}

