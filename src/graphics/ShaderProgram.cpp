#include "ShaderProgram.h"

#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace fluidlab::graphics {
namespace {

[[nodiscard]] std::string readTextFile(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw std::runtime_error("Failed to open shader file: " + path.string());
    }

    std::ostringstream contents;
    contents << input.rdbuf();
    if (!input.good() && !input.eof()) {
        throw std::runtime_error("Failed while reading shader file: " + path.string());
    }
    return contents.str();
}

[[nodiscard]] GLuint compileShader(GLenum type, const std::filesystem::path& path) {
    const std::string source = readTextFile(path);
    const char* sourcePointer = source.c_str();

    const GLuint shader = glCreateShader(type);
    if (shader == 0) {
        throw std::runtime_error("OpenGL failed to create a shader object for: " + path.string());
    }

    glShaderSource(shader, 1, &sourcePointer, nullptr);
    glCompileShader(shader);

    GLint compileSucceeded = GL_FALSE;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &compileSucceeded);

    GLint logLength = 0;
    glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &logLength);
    if (logLength > 1) {
        std::vector<char> log(static_cast<std::size_t>(logLength));
        glGetShaderInfoLog(shader, logLength, nullptr, log.data());
        if (compileSucceeded == GL_FALSE) {
            const std::string message = "Shader compilation failed for " + path.string() + ":\n" + log.data();
            glDeleteShader(shader);
            throw std::runtime_error(message);
        }
    }

    if (compileSucceeded == GL_FALSE) {
        glDeleteShader(shader);
        throw std::runtime_error("Shader compilation failed without a driver log: " + path.string());
    }

    return shader;
}

[[nodiscard]] GLuint linkProgram(const std::vector<GLuint>& shaders) {
    const GLuint program = glCreateProgram();
    if (program == 0) {
        throw std::runtime_error("OpenGL failed to create a program object.");
    }

    for (const GLuint shader : shaders) {
        glAttachShader(program, shader);
    }
    glLinkProgram(program);

    GLint linkSucceeded = GL_FALSE;
    glGetProgramiv(program, GL_LINK_STATUS, &linkSucceeded);

    GLint logLength = 0;
    glGetProgramiv(program, GL_INFO_LOG_LENGTH, &logLength);
    if (logLength > 1) {
        std::vector<char> log(static_cast<std::size_t>(logLength));
        glGetProgramInfoLog(program, logLength, nullptr, log.data());
        if (linkSucceeded == GL_FALSE) {
            const std::string message = "Shader program linking failed:\n" + std::string(log.data());
            glDeleteProgram(program);
            throw std::runtime_error(message);
        }
    }

    if (linkSucceeded == GL_FALSE) {
        glDeleteProgram(program);
        throw std::runtime_error("Shader program linking failed without a driver log.");
    }

    for (const GLuint shader : shaders) {
        glDetachShader(program, shader);
    }
    return program;
}

} // namespace

ShaderProgram::ShaderProgram(GLuint handle) noexcept
    : m_handle(handle) {
}

ShaderProgram::~ShaderProgram() {
    if (m_handle != 0) {
        glDeleteProgram(m_handle);
    }
}

ShaderProgram::ShaderProgram(ShaderProgram&& other) noexcept
    : m_handle(std::exchange(other.m_handle, 0)) {
}

ShaderProgram& ShaderProgram::operator=(ShaderProgram&& other) noexcept {
    if (this != &other) {
        if (m_handle != 0) {
            glDeleteProgram(m_handle);
        }
        m_handle = std::exchange(other.m_handle, 0);
    }
    return *this;
}

ShaderProgram ShaderProgram::fromComputeFile(const std::filesystem::path& computePath) {
    const GLuint computeShader = compileShader(GL_COMPUTE_SHADER, computePath);
    try {
        const GLuint program = linkProgram({computeShader});
        glDeleteShader(computeShader);
        return ShaderProgram(program);
    } catch (...) {
        glDeleteShader(computeShader);
        throw;
    }
}

ShaderProgram ShaderProgram::fromGraphicsFiles(
    const std::filesystem::path& vertexPath,
    const std::filesystem::path& fragmentPath) {
    const GLuint vertexShader = compileShader(GL_VERTEX_SHADER, vertexPath);
    GLuint fragmentShader = 0;
    try {
        fragmentShader = compileShader(GL_FRAGMENT_SHADER, fragmentPath);
        const GLuint program = linkProgram({vertexShader, fragmentShader});
        glDeleteShader(vertexShader);
        glDeleteShader(fragmentShader);
        return ShaderProgram(program);
    } catch (...) {
        glDeleteShader(vertexShader);
        if (fragmentShader != 0) {
            glDeleteShader(fragmentShader);
        }
        throw;
    }
}

void ShaderProgram::use() const noexcept {
    glUseProgram(m_handle);
}

GLuint ShaderProgram::handle() const noexcept {
    return m_handle;
}

GLint ShaderProgram::uniformLocation(std::string_view name) const {
    const std::string nullTerminatedName(name);
    const GLint location = glGetUniformLocation(m_handle, nullTerminatedName.c_str());
    if (location < 0) {
        throw std::runtime_error("Required shader uniform was not found: " + nullTerminatedName);
    }
    return location;
}

} // namespace fluidlab::graphics
