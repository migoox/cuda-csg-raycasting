#include "textures.hpp"
#include "gl_debug.h"
#include <iostream>
#define STB_IMAGE_IMPLEMENTATION
#include "vendor/stbi_image.h"
#include <cmath>

using namespace renderer;

Image::Image(uint32_t width, uint32_t height, uint32_t color)
: m_width(width), m_height(height), m_bpp(4), m_path(std::nullopt) {
    m_data = std::vector<uint32_t>(m_width * m_height, color);
}

void swap_endianess(uint32_t *pixel) {
    *pixel = ((*pixel & 0xFF000000) >> 24) |
             ((*pixel & 0x00FF0000) >> 8)  |
             ((*pixel & 0x0000FF00) << 8)  |
             ((*pixel & 0x000000FF) << 24);
}

Image::Image(const std::string& path)
: m_path(path) {
    stbi_set_flip_vertically_on_load(1);
    int width, height, bpp;
    auto buff = reinterpret_cast<uint32_t*>(stbi_load(path.c_str(), &width, &height, &bpp, 4));

    for (int i = 0; i < width * height; ++i) {
        swap_endianess(&buff[i]);
    }
    m_width = width;
    m_height = height;
    m_bpp = bpp;

    if (!buff) {
        std::cerr << "[STBI]: " << "Couldn't load texture " << path << "\n";
        return;
    }

    m_data.insert(m_data.end(), &buff[0], &buff[m_width * m_height]);
    stbi_image_free(buff);
}

void Image::clear(uint32_t color) {
    for (auto& pixel : m_data) {
        pixel = color;
    }
}

void Image::set_pixel_safe(uint32_t x, uint32_t y, uint32_t color) {
    if (!is_in_bounds(x, y)) {
        return;
    }

    m_data[x + y * m_width] = color;
}

void Image::set_pixel(uint32_t x, uint32_t y, uint32_t color) {
    m_data[x + y * m_width] = color;
}

void Image::resize(uint32_t new_width, uint32_t new_height, uint32_t color) {
    m_data.resize(new_width * new_height, color);
    m_width = new_width;
    m_height = new_height;
}

bool Image::is_in_bounds(uint32_t x, uint32_t y) const {
    return x < m_width && y < m_height;
}

uint32_t Image::get_pixel(uint32_t x, uint32_t y) const {
    return m_data[x + y * m_width];
}

namespace renderer {
    uint32_t get_color_rgb_norm(float r, float g, float b) {
        return static_cast<uint32_t>(std::round(r * 255.f)) << 24 |
               static_cast<uint32_t>(std::round(g * 255.f)) << 16 |
               static_cast<uint32_t>(std::round(b * 255.f)) << 8 |
               0xFF;
    }

    uint32_t get_color_rgb(uint8_t r, uint8_t g, uint8_t b) {
        return static_cast<uint32_t>(r) << 24 |
               static_cast<uint32_t>(g) << 16 |
               static_cast<uint32_t>(b) << 8 |
               0xFF;
    }

    uint32_t get_color_rgba_norm(float r, float g, float b, float a) {
        return static_cast<uint32_t>(std::round(r * 255.f)) << 24 |
               static_cast<uint32_t>(std::round(g * 255.f)) << 16 |
               static_cast<uint32_t>(std::round(b * 255.f)) << 8 |
               static_cast<uint32_t>(std::round(a * 255.f));
    }

    uint32_t get_color_rgba(uint8_t r, uint8_t g, uint8_t b, uint8_t a) {
        return static_cast<uint32_t>(r) << 24 |
               static_cast<uint32_t>(g) << 16 |
               static_cast<uint32_t>(b) << 8 |
               static_cast<uint32_t>(a);
    }

    float get_red_norm(uint32_t color) {
        // Extract the red component from the uint32_t color and normalize it to the range [0, 1]
        return static_cast<float>((color >> 24) & 0xFF) / 255.f;
    }

    uint8_t get_red(uint32_t color) {
        // Extract and return the red component as an 8-bit unsigned integer
        return static_cast<uint8_t>((color >> 24) & 0xFF);
    }

    float get_green_norm(uint32_t color) {
        // Extract the green component from the uint32_t color and normalize it to the range [0, 1]
        return static_cast<float>((color >> 16) & 0xFF) / 255.f;
    }

    uint8_t get_green(uint32_t color) {
        // Extract and return the green component as an 8-bit unsigned integer
        return static_cast<uint8_t>((color >> 16) & 0xFF);
    }

    float get_blue_norm(uint32_t color) {
        // Extract the blue component from the uint32_t color and normalize it to the range [0, 1]
        return static_cast<float>((color >> 8) & 0xFF) / 255.f;
    }

    uint8_t get_blue(uint32_t color) {
        // Extract and return the blue component as an 8-bit unsigned integer
        return static_cast<uint8_t>((color >> 8) & 0xFF);
    }

    float get_alpha_norm(uint32_t color) {
        // Extract the alpha component from the uint32_t color and normalize it to the range [0, 1]
        return static_cast<float>(color & 0xFF) / 255.f;
    }

    uint8_t get_alpha(uint32_t color) {
        // Extract and return the alpha component as an 8-bit unsigned integer
        return static_cast<uint8_t>(color & 0xFF);
    }
}
TextureResource::TextureResource(const std::string& path, bool mipmap)
: m_gl_id(0), m_width(0), m_height(0), m_bpp(0), m_mipmap(mipmap) {
    stbi_set_flip_vertically_on_load(1);
    auto buff = stbi_load(path.c_str(), &m_width, &m_height, &m_bpp, 4);

    if (!buff) {
        std::cerr << "[STBI]: " << "Couldn't load texture " << path << "\n";
        return;
    }

    GLCall( glGenTextures(1, &m_gl_id) );
    GLCall( glBindTexture(GL_TEXTURE_2D, m_gl_id) );

    GLCall( glTexImage2D(
            GL_TEXTURE_2D,
            0,
            GL_RGBA8,
            m_width,
            m_height,
            0,
            GL_RGBA,
            GL_UNSIGNED_BYTE,
            buff
    ) );

    if (mipmap) {
        GLCall( glGenerateMipmap(GL_TEXTURE_2D) );
    }

    stbi_image_free(buff);
}

TextureResource::TextureResource(const Image &img, bool mipmap)
: m_gl_id(0), m_width(img.get_width()), m_height(img.get_height()), m_bpp(4), m_mipmap(mipmap) {

    GLCall( glGenTextures(1, &m_gl_id) );
    GLCall( glBindTexture(GL_TEXTURE_2D, m_gl_id) );

    GLCall( glTexImage2D(
            GL_TEXTURE_2D,
            0,
            GL_RGBA,
            m_width,
            m_height,
            0,
            GL_RGBA,
            GL_UNSIGNED_INT_8_8_8_8,
            img.raw()
    ) );

    if (mipmap) {
        GLCall( glGenerateMipmap(GL_TEXTURE_2D) );
    }
}

TextureResource::~TextureResource() {
    if (m_gl_id > 0) {
        GLCall( glDeleteTextures(1, &m_gl_id) );
    }
}

void TextureResource::bind(uint32_t unit) const {
    GLCall( glActiveTexture(GL_TEXTURE0 + unit) );
    GLCall( glBindTexture(GL_TEXTURE_2D, m_gl_id) );
}

void TextureResource::unbind() const {
    GLCall( glBindTexture(GL_TEXTURE_2D, 0) );
}

TextureResource::TextureResource(TextureResource &&other) noexcept
: m_gl_id(other.m_gl_id), m_width(other.m_width), m_height(other.m_height),
m_bpp(other.m_bpp), m_mipmap(other.m_mipmap) {
    other.m_gl_id = 0;
}

void TextureResource::rebuild(const Image &img, bool mipmap) {
    m_width = img.get_width();
    m_height = img.get_height();
    m_mipmap = mipmap;
    GLCall( glBindTexture(GL_TEXTURE_2D, m_gl_id) );
    GLCall( glTexImage2D(
            GL_TEXTURE_2D,
            0,
            GL_RGBA,
            m_width,
            m_height,
            0,
            GL_RGBA,
            GL_UNSIGNED_INT_8_8_8_8,
            img.raw()
    ) );

    if (mipmap) {
        GLCall( glGenerateMipmap(GL_TEXTURE_2D) );
    }
}

void TextureResource::update(const Image &img) {
    GLCall( glBindTexture(GL_TEXTURE_2D, m_gl_id) );
    GLCall( glTexSubImage2D(
            GL_TEXTURE_2D,
            0,
            0,
            0,
            img.get_width(),
            img.get_height(),
            GL_RGBA,
            GL_UNSIGNED_INT_8_8_8_8,
            img.raw()
    ) );
}

void TextureResource::update(int x, int y, const Image &img) {
    GLCall( glBindTexture(GL_TEXTURE_2D, m_gl_id) );
    GLCall( glTexSubImage2D(
            GL_TEXTURE_2D,
            0,
            x,
            y,
            img.get_width(),
            img.get_height(),
            GL_RGBA,
            GL_UNSIGNED_INT_8_8_8_8,
            img.raw()
    ) );
}

Texture::Texture() : m_resource(std::nullopt) { }
Texture::Texture(const std::shared_ptr<const TextureResource>& resource) : m_resource(resource) { }

void Texture::bind(uint32_t unit) const {
    if (!m_resource.has_value()) {
        return;
    }

    m_resource->get()->bind(unit);

    if (m_resource->get()->is_mipmap()) {
        GLCall( glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR) );
        GLCall( glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR) );
        GLCall( glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE) );
        GLCall( glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE) );
    } else {
        if (this->filtering == TextureFiltering::Linear) {
            GLCall( glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR) );
            GLCall( glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR) );
        } else {
            GLCall( glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST) );
            GLCall( glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST) );
        }

        if (this->wrapping == TextureWrapping::Repeat) {
            GLCall( glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT) );
            GLCall( glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT) );
        } else if (this->wrapping == TextureWrapping::ClampToEdge) {
            GLCall( glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE) );
            GLCall( glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE) );
        } else if (this->wrapping == TextureWrapping::ClampToBorder) {
            GLCall( glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER) );
            GLCall( glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER) );
        } else {
            GLCall( glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_MIRRORED_REPEAT) );
            GLCall( glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_MIRRORED_REPEAT) );
        }
    }
}

void Texture::clear_res() {
    m_resource.reset();
}

bool Texture::has_res() const {
    return m_resource.has_value();
}

void Texture::set_res(const std::shared_ptr<const TextureResource>& resource) {
    m_resource.emplace(resource);
}

