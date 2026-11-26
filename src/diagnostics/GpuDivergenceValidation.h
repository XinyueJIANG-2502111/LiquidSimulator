#pragma once

#include <filesystem>
#include <iosfwd>

namespace fluidlab::diagnostics {

int runGpuDivergenceValidation(
    std::ostream& output,
    const std::filesystem::path& divergenceShaderPath);

} // namespace fluidlab::diagnostics
