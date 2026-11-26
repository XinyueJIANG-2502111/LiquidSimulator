#version 460 core

layout(std430, binding = 0) readonly buffer ParticleBuffer { vec4 particle[]; };

uniform vec2 domainSize;

void main() {
    const vec4 value = particle[gl_VertexID];
    if (value.z < 0.5 || any(isnan(value.xy)) || any(isinf(value.xy))) {
        gl_Position = vec4(2.0, 2.0, 0.0, 1.0);
        return;
    }
    const vec2 normalizedPosition = value.xy / domainSize;
    gl_Position = vec4(normalizedPosition * 2.0 - 1.0, 0.0, 1.0);
}
