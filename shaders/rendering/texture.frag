#version 460 core

layout(binding = 0) uniform sampler2D u_sourceTexture;

layout(location = 0) in vec2 v_uv;
layout(location = 0) out vec4 o_color;

uniform bool u_desktopStyleEnabled;
uniform bool u_gridEnabled;
uniform float u_waterAlpha;
uniform float u_gridStrength;
uniform float u_gridCellSpacing;
uniform vec3 u_backgroundColor;
uniform vec3 u_gridColor;
uniform vec3 u_waterColor;

float gridCoverage(vec2 cellCoordinate) {
    vec2 gridCoordinate = cellCoordinate / max(u_gridCellSpacing, 1.0);
    vec2 phase = abs(fract(gridCoordinate + 0.5) - 0.5);
    vec2 pixelWidth = max(fwidth(gridCoordinate), vec2(1.0e-6));
    vec2 line = 1.0 - smoothstep(pixelWidth * 0.35, pixelWidth * 1.15, phase);
    return clamp(max(line.x, line.y), 0.0, 1.0);
}

void main() {
    vec4 source = texture(u_sourceTexture, v_uv);
    if (!u_desktopStyleEnabled) {
        o_color = source;
        return;
    }

    vec3 background = u_backgroundColor;
    if (u_gridEnabled) {
        vec2 cells = vec2(textureSize(u_sourceTexture, 0));
        float coverage = gridCoverage(v_uv * cells);
        background = mix(background, u_gridColor,
            coverage * clamp(u_gridStrength, 0.0, 1.0));
    }

    // The classification visualization uses distinct, non-overlapping colors.
    // Compositing this single occupancy texture avoids particle-overlap alpha buildup.
    bool solidCell = source.r > 0.25 && source.g > 0.25 && source.b < 0.60;
    bool fluidCell = source.b > 0.65 && source.b > source.r * 1.5;
    if (solidCell) {
        o_color = vec4(source.rgb, 1.0);
    } else if (fluidCell) {
        float alpha = clamp(u_waterAlpha, 0.0, 1.0);
        o_color = vec4(mix(background, u_waterColor, alpha), 1.0);
    } else {
        o_color = vec4(background, 1.0);
    }
}
