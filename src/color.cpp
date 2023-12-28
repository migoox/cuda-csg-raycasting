#include "color.hpp"
#include <cmath>

uint32_t color::get_color_rgb_norm(float r, float g, float b) {
    return static_cast<uint32_t>(std::round(r * 255.f)) << 24 |
           static_cast<uint32_t>(std::round(g * 255.f)) << 16 |
           static_cast<uint32_t>(std::round(b * 255.f)) << 8 |
           0xFF;
}

uint32_t color::get_color_rgb(uint8_t r, uint8_t g, uint8_t b) {
    return static_cast<uint32_t>(r) << 24 |
           static_cast<uint32_t>(g) << 16 |
           static_cast<uint32_t>(b) << 8 |
           0xFF;
}

uint32_t color::get_color_rgba_norm(float r, float g, float b, float a) {
    return static_cast<uint32_t>(std::round(r * 255.f)) << 24 |
           static_cast<uint32_t>(std::round(g * 255.f)) << 16 |
           static_cast<uint32_t>(std::round(b * 255.f)) << 8 |
           static_cast<uint32_t>(std::round(a * 255.f));
}

uint32_t color::get_color_rgba(uint8_t r, uint8_t g, uint8_t b, uint8_t a) {
    return static_cast<uint32_t>(r) << 24 |
           static_cast<uint32_t>(g) << 16 |
           static_cast<uint32_t>(b) << 8 |
           static_cast<uint32_t>(a);
}

float color::get_red_norm(uint32_t color) {
    // Extract the red component from the uint32_t color and normalize it to the range [0, 1]
    return static_cast<float>((color >> 24) & 0xFF) / 255.f;
}

uint8_t color::get_red(uint32_t color) {
    // Extract and return the red component as an 8-bit unsigned integer
    return static_cast<uint8_t>((color >> 24) & 0xFF);
}

float color::get_green_norm(uint32_t color) {
    // Extract the green component from the uint32_t color and normalize it to the range [0, 1]
    return static_cast<float>((color >> 16) & 0xFF) / 255.f;
}

uint8_t color::get_green(uint32_t color) {
    // Extract and return the green component as an 8-bit unsigned integer
    return static_cast<uint8_t>((color >> 16) & 0xFF);
}

float color::get_blue_norm(uint32_t color) {
    // Extract the blue component from the uint32_t color and normalize it to the range [0, 1]
    return static_cast<float>((color >> 8) & 0xFF) / 255.f;
}

uint8_t color::get_blue(uint32_t color) {
    // Extract and return the blue component as an 8-bit unsigned integer
    return static_cast<uint8_t>((color >> 8) & 0xFF);
}

float color::get_alpha_norm(uint32_t color) {
    // Extract the alpha component from the uint32_t color and normalize it to the range [0, 1]
    return static_cast<float>(color & 0xFF) / 255.f;
}

uint8_t color::get_alpha(uint32_t color) {
    // Extract and return the alpha component as an 8-bit unsigned integer
    return static_cast<uint8_t>(color & 0xFF);
}