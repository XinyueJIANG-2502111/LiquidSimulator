#pragma once

#include <filesystem>
#include <string_view>

#include <glad/gl.h>

namespace fluidlab::graphics {

class ShaderProgram final {
public:
    ShaderProgram() = default;
    explicit ShaderProgram(GLuint handle) noexcept;
    ~ShaderProgram();

    ShaderProgram(const ShaderProgram&) = delete;
    ShaderProgram& operator=(const ShaderProgram&) = delete;

    ShaderProgram(ShaderProgram&& other) noexcept;
    ShaderProgram& operator=(ShaderProgram&& other) noexcept;

    [[nodiscard]] static ShaderProgram fromComputeFile(const std::filesystem::path& computePath);
    [[nodiscard]] static ShaderProgram fromGraphicsFiles(
        const std::filesystem::path& vertexPath,
        const std::filesystem::path& fragmentPath);

    void use() const noexcept;
    [[nodiscard]] GLuint handle() const noexcept;
    [[nodiscard]] GLint uniformLocation(std::string_view name) const;

private:
    GLuint m_handle = 0;
};

} // namespace fluidlab::graphics
