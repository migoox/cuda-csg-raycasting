#ifndef OPENGL_3D_SCENERY_RESOURCE_MANAGER_H
#define OPENGL_3D_SCENERY_RESOURCE_MANAGER_H
#include <cstdint>
#include <unordered_map>
#include <memory>
#include <string>
#include <GL/glew.h>
#include <optional>
#include <filesystem>
#include <vector>

namespace renderer {
    class Image {
    public:
        explicit Image(uint32_t width, uint32_t height, uint32_t color = 0x000000FF);
        explicit Image(const std::string& path);

        bool is_in_bounds(uint32_t x, uint32_t y) const;

        void set_pixel(uint32_t x, uint32_t y, uint32_t color);
        void set_pixel_safe(uint32_t x, uint32_t y, uint32_t color);

        uint32_t get_pixel(uint32_t x, uint32_t y) const;

        void resize(uint32_t new_width, uint32_t new_height, uint32_t color = 0x000000FF);
        void clear(uint32_t color);

        uint32_t get_width() const { return m_width; }
        uint32_t get_height() const { return m_height; }

        std::optional<std::string_view> get_path() const { return m_path; }

        const uint32_t* raw() const { return m_data.data(); }
        const uint8_t* raw_as_bytes() const { return reinterpret_cast<const uint8_t*>(m_data.data()); }

    private:
        uint32_t m_width, m_height, m_bpp;
        std::vector<uint32_t> m_data;

        std::optional<std::string> m_path;
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

    class TextureResource {
    public:
        TextureResource() = delete;
        explicit TextureResource(const std::string& path, bool mipmap = false);
        explicit TextureResource(const Image& img, bool mipmap = false);

        TextureResource(TextureResource&& other) noexcept;
        TextureResource(const TextureResource& other) = delete;
        TextureResource& operator=(TextureResource&& other) = delete;
        TextureResource& operator=(const TextureResource& other) = delete;

        ~TextureResource();

        void rebuild(const Image& img, bool mipmap = false);
        void update(const Image& img);
        void update(int x, int y, const Image& img);

        bool is_mipmap() const { return m_mipmap; }

        void bind(uint32_t unit) const;
        void unbind() const;

    private:
        GLuint m_gl_id;
        int m_width;
        int m_height;
        int m_bpp;
        bool m_mipmap;
    };

    enum class TextureFiltering {
        Linear,
        Nearest
    };

    enum class TextureWrapping {
        Repeat,
        MirroredRepeat,
        ClampToEdge,
        ClampToBorder,
    };

    class Texture {
    public:
        Texture();
        Texture(const std::shared_ptr<const TextureResource>& resouce);

        TextureFiltering filtering = TextureFiltering::Linear;
        TextureWrapping wrapping = TextureWrapping::Repeat;

        void clear_res();
        bool has_res() const;
        void set_res(const std::shared_ptr<const TextureResource>& resource);

        void bind(uint32_t unit) const;

    private:
        std::optional<std::shared_ptr<const TextureResource>> m_resource;
    };
}

#endif //OPENGL_3D_SCENERY_RESOURCE_MANAGER_H
