#version 460 core

layout(location = 0) out vec2 v_uv;

void main() {
    // A fullscreen triangle avoids vertex buffers and covers the viewport without
    // a diagonal interpolation seam.
    const vec2 position = vec2(
        float((gl_VertexID << 1) & 2),
        float(gl_VertexID & 2));
    v_uv = position;
    gl_Position = vec4(position * 2.0 - 1.0, 0.0, 1.0);
}
