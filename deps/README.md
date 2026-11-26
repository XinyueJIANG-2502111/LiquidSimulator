# Phase 0 dependencies

All third-party headers, libraries, generated loader code, and the Visual Studio
property sheet used by this project live under `deps/`. No system-wide install is
required.

## GLFW 3.4

- Purpose: creates the Win32 window and OpenGL context and handles basic input.
- Package: official `glfw-3.4.bin.WIN64.zip`; only the headers, VS2022 libraries,
  upstream README, and license are retained in `deps/glfw`.
- Link mode: `lib-vc2022/glfw3dll.lib`; `glfw3.dll` is copied beside the
  executable. The DLL avoids mixing GLFW's release C runtime with the project's
  Debug runtime.
- License: zlib/libpng; see `glfw/LICENSE.md`.
- Source: <https://github.com/glfw/glfw/releases/tag/3.4>

## GLAD 2.0.8

- Purpose: loads the OpenGL 4.6 Core function entry points after GLFW creates a
  context.
- Configuration: C generator, `gl:core=4.6`, no extensions explicitly selected,
  reproducible/offline Khronos specification snapshot.
- Generated files: `glad/include/glad/gl.h`, `glad/include/KHR/khrplatform.h`, and
  `glad/src/gl.c`.
- Generated-code licensing: public domain, WTFPL, or CC0 for GLAD output; the
  Khronos header carries its own notice in the generated file.
- Source: <https://github.com/Dav1dde/glad/tree/glad2>

## Visual Studio configuration

`Dependencies.props` is imported by `LiquidSimulator.vcxproj`. It adds the GLFW
and GLAD include directories and links GLFW plus the required Windows OpenGL and
window-system libraries. The project compiles GLAD's generated C source directly.

Only the x64 Debug and Release configurations are supported by this dependency
layout.
