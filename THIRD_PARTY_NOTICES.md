# Third-Party Notices

LiquidSimulator includes or links the following third-party components.

## GLFW 3.4

- Purpose: Windows window creation, OpenGL context creation, and input handling.
- Distribution: bundled headers, VS2022 import library, and `glfw3.dll`.
- License: zlib/libpng license.
- Full license text: `licenses/GLFW-LICENSE.md` in the portfolio package.
- Project: https://www.glfw.org/

## GLAD 2.0.8 generated loader

- Purpose: OpenGL 4.6 Core function loading.
- Distribution: generated C source and headers compiled into LiquidSimulator.
- Generated output is provided under public-domain, WTFPL, or CC0 terms as described by GLAD.
- The bundled Khronos platform header retains its own copyright and permission notice.
- Project: https://github.com/Dav1dde/glad

No third-party component is downloaded during the CMake configure or build process.

## Microsoft Visual C++ Redistributable Runtime

- Purpose: C and C++ runtime required by the MSVC-built executable and GLFW DLL.
- Distribution: redistributable runtime DLLs are installed into the portfolio ZIP by CMake's `InstallRequiredSystemLibraries` module.
- License: Microsoft Visual Studio 2022 license terms and redistributable code provisions.
