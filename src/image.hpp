#ifndef CSG_RAY_TRACING_IMAGE_HPP
#define CSG_RAY_TRACING_IMAGE_HPP
#include <cstdint>
#include <vector>

namespace img {
    class Image {
    public:
        Image(uint32_t width, uint32_t height);

        bool in_bounds(uint32_t x, uint32_t y) const;

        void set_pixel(uint32_t x, uint32_t y, uint32_t color);
        void set_pixel_safe(uint32_t x, uint32_t y, uint32_t color);

        uint32_t get_pixel(uint32_t x, uint32_t y) const;

        void resize(uint32_t new_width, uint32_t new_height);
        void clear(uint32_t color);

        uint32_t get_width() const { return m_width; }
        uint32_t get_height() const { return m_height; }

        const uint32_t* raw() const { return m_data.data(); }
    private:
        uint32_t m_width, m_height;
        std::vector<uint32_t> m_data;
    };

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

#endif //CSG_RAY_TRACING_IMAGE_HPP
