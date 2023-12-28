#ifndef CSG_RAY_TRACING_COLOR_HPP
#define CSG_RAY_TRACING_COLOR_HPP
#include <cstdint>

namespace color {
    uint32_t get_color_rgb_norm(float r, float g, float b);
    uint32_t get_color_rgb(uint8_t r, uint8_t g, uint8_t b);

    uint32_t get_color_rgba_norm(float r, float g, float b, float a);
    uint32_t get_color_rgba(uint8_t r, uint8_t g, uint8_t b, uint8_t a);

    float get_red_norm(uint32_t color);
    uint8_t get_red(uint32_t color);

    float get_green_norm(uint32_t color);
    uint8_t get_green(uint32_t color);

    float get_blue_norm(uint32_t color);
    uint8_t get_blue(uint32_t color);

    float get_alpha_norm(uint32_t color);
    uint8_t get_alpha(uint32_t color);
}

#endif //CSG_RAY_TRACING_COLOR_HPP
