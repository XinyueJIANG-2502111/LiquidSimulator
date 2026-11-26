cmake_minimum_required(VERSION 3.24)

project(LiquidSimulator VERSION 2.0.0 LANGUAGES C CXX)

if(NOT WIN32)
    message(FATAL_ERROR "LiquidSimulator Desktop currently supports Windows only.")
endif()

if(NOT CMAKE_SIZEOF_VOID_P EQUAL 8)
    message(FATAL_ERROR "LiquidSimulator requires an x64 CMake generator.")
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(LIQUIDSIMULATOR_DESKTOP_RELEASE "Build only the interactive desktop demo" OFF)

set(LIQUIDSIMULATOR_SOURCES
    deps/glad/src/gl.c
    src/desktop/ContainerMotionInput.cpp
    src/desktop/DesktopPerformanceMetrics.cpp
    src/desktop/DesktopSettledState.cpp
    src/desktop/WindowsMoveLoopAdapter.cpp
    src/graphics/ShaderProgram.cpp
    src/main.cpp
    src/simulation/SimulationStepInput.cpp
    src/simulation/gpu/GpuAsyncDynamicFreeSurfaceStepper.cpp
    src/simulation/gpu/GpuDynamicFreeSurfaceStepper.cpp
    src/simulation/gpu/GpuMacGrid.cpp
    src/simulation/mac/DynamicFreeSurfaceStepper.cpp
    src/simulation/mac/DynamicLevelSet.cpp
    src/simulation/mac/FluidStepper.cpp
    src/simulation/mac/FreeSurfacePressureProjection.cpp
    src/simulation/mac/GhostFluidPressureProjection.cpp
    src/simulation/mac/InterfaceGeometry.cpp
    src/simulation/mac/LevelSetInterface.cpp
    src/simulation/mac/MacGrid.cpp
    src/simulation/mac/MarkerParticles.cpp
    src/simulation/mac/PressureProjection.cpp
    src/simulation/mac/Scenarios.cpp
    src/simulation/mac/VelocityAdvection.cpp
    src/simulation/mac/VelocityExtrapolation.cpp
    src/visualization/FieldVisualization.cpp
)

if(NOT LIQUIDSIMULATOR_DESKTOP_RELEASE)
    list(APPEND LIQUIDSIMULATOR_SOURCES
        src/diagnostics/CpuBenchmark.cpp
        src/diagnostics/DesktopInteractionValidation.cpp
        src/diagnostics/DynamicBenchmark.cpp
        src/diagnostics/GpuDivergenceValidation.cpp
        src/diagnostics/GpuLevelSetValidation.cpp
        src/diagnostics/GpuMacValidation.cpp
        src/diagnostics/GpuVofValidation.cpp
        src/diagnostics/InterfaceValidation.cpp
        src/diagnostics/LevelSetDynamicValidation.cpp
        src/diagnostics/VofValidation.cpp
        src/simulation/gpu/GpuVolumeFraction.cpp
        src/simulation/mac/CoupledLevelSetVOF.cpp
        src/simulation/mac/VolumeFractionField.cpp
        src/tests/Phase1Tests.cpp
    )
endif()

add_executable(LiquidSimulator ${LIQUIDSIMULATOR_SOURCES})

if(LIQUIDSIMULATOR_DESKTOP_RELEASE)
    set_target_properties(LiquidSimulator PROPERTIES WIN32_EXECUTABLE TRUE)
    target_compile_definitions(LiquidSimulator PRIVATE LIQUIDSIMULATOR_DESKTOP_RELEASE)
    target_link_libraries(LiquidSimulator PRIVATE user32)
endif()

target_include_directories(LiquidSimulator PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/deps/glad/include"
    "${CMAKE_CURRENT_SOURCE_DIR}/deps/glfw/include"
)

target_compile_definitions(LiquidSimulator PRIVATE
    GLFW_DLL
    UNICODE
    _UNICODE
)

if(MSVC)
    target_compile_options(LiquidSimulator PRIVATE /W4 /utf-8 /permissive-)
    target_compile_options(LiquidSimulator PRIVATE $<$<CONFIG:Release>:/O2>)
else()
    message(FATAL_ERROR "The bundled GLFW import library requires MSVC and x64 Windows.")
endif()

add_library(LiquidSimulatorGlfw SHARED IMPORTED GLOBAL)
set_target_properties(LiquidSimulatorGlfw PROPERTIES
    IMPORTED_IMPLIB "${CMAKE_CURRENT_SOURCE_DIR}/deps/glfw/lib-vc2022/glfw3dll.lib"
    IMPORTED_LOCATION "${CMAKE_CURRENT_SOURCE_DIR}/deps/glfw/lib-vc2022/glfw3.dll"
    INTERFACE_INCLUDE_DIRECTORIES "${CMAKE_CURRENT_SOURCE_DIR}/deps/glfw/include"
)

find_package(OpenGL REQUIRED)
target_link_libraries(LiquidSimulator PRIVATE LiquidSimulatorGlfw OpenGL::GL)

source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}" FILES ${LIQUIDSIMULATOR_SOURCES})

add_custom_command(TARGET LiquidSimulator POST_BUILD
    COMMAND "${CMAKE_COMMAND}" -E copy_if_different
        "${CMAKE_CURRENT_SOURCE_DIR}/deps/glfw/lib-vc2022/glfw3.dll"
        "$<TARGET_FILE_DIR:LiquidSimulator>/glfw3.dll"
    COMMAND "${CMAKE_COMMAND}" -E copy_directory
        "${CMAKE_CURRENT_SOURCE_DIR}/shaders"
        "$<TARGET_FILE_DIR:LiquidSimulator>/shaders"
    COMMAND "${CMAKE_COMMAND}" -E make_directory
        "$<TARGET_FILE_DIR:LiquidSimulator>/assets/desktop"
    COMMAND "${CMAKE_COMMAND}" -E copy_if_different
        "${CMAKE_CURRENT_SOURCE_DIR}/assets/desktop/settled_64x48.lsstate"
        "$<TARGET_FILE_DIR:LiquidSimulator>/assets/desktop/settled_64x48.lsstate"
    COMMENT "Copying LiquidSimulator runtime resources"
)

install(TARGETS LiquidSimulator RUNTIME DESTINATION .)
install(FILES
    "${CMAKE_CURRENT_SOURCE_DIR}/deps/glfw/lib-vc2022/glfw3.dll"
    DESTINATION .
)
if(LIQUIDSIMULATOR_DESKTOP_RELEASE)
    # GpuMacGrid eagerly loads its MAC shaders during initialization.
    install(DIRECTORY shaders/compute/ DESTINATION shaders/compute
        FILES_MATCHING PATTERN "mac_*.comp")
    install(FILES shaders/rendering/fullscreen.vert shaders/rendering/texture.frag
        DESTINATION shaders/rendering)
    install(FILES docs/desktop-release-quickstart.txt DESTINATION . RENAME README.txt)
else()
    install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/shaders" DESTINATION .)
    install(FILES README.md PORTFOLIO_README.md DESTINATION .)
endif()
install(FILES
    "${CMAKE_CURRENT_SOURCE_DIR}/assets/desktop/settled_64x48.lsstate"
    DESTINATION assets/desktop
)
install(FILES
    "${CMAKE_CURRENT_SOURCE_DIR}/THIRD_PARTY_NOTICES.md"
    DESTINATION .
)
install(FILES
    "${CMAKE_CURRENT_SOURCE_DIR}/deps/glfw/LICENSE.md"
    DESTINATION licenses
    RENAME GLFW-LICENSE.md
)

# Bundle the redistributable MSVC runtime so the portfolio ZIP does not require
# Visual Studio to be installed on the target machine.
set(CMAKE_INSTALL_SYSTEM_RUNTIME_DESTINATION .)
include(InstallRequiredSystemLibraries)

set(CPACK_GENERATOR "ZIP")
set(CPACK_PACKAGE_NAME "LiquidSimulator")
set(CPACK_PACKAGE_VENDOR "LiquidSimulator Project")
set(CPACK_PACKAGE_VERSION "${PROJECT_VERSION}")
if(LIQUIDSIMULATOR_DESKTOP_RELEASE)
    set(CPACK_PACKAGE_FILE_NAME "LiquidSimulator-${PROJECT_VERSION}-Desktop-Release-Windows-x64")
else()
    set(CPACK_PACKAGE_FILE_NAME "LiquidSimulator-${PROJECT_VERSION}-Portfolio-Windows-x64")
endif()
set(CPACK_PACKAGE_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/dist")
set(CPACK_INCLUDE_TOPLEVEL_DIRECTORY ON)
set(CPACK_MONOLITHIC_INSTALL ON)
include(CPack)
