#version 330 core
out vec4 FragColor;

in vec2 TexCoords;

uniform sampler2D screenTexture;

// MAXIMUM BRIGHTNESS post-processing parameters
const float exposure = 3.0;         // ULTRA high exposure
const float contrast = 2.0;         // MAXIMUM contrast
const float saturation = 2.5;       // MAXIMUM color saturation
const float gamma = 1.8;            // Lower gamma for even more brightness
const float bloomStrength = 1.0;    // MAXIMUM bloom effect
const bool vignette = false;        // Disable vignette to avoid darkening edges

vec3 adjustExposure(vec3 color, float exposure) {
    return vec3(1.0) - exp(-color * exposure);
}

vec3 adjustContrast(vec3 color, float contrast) {
    return 0.5 + contrast * (color - 0.5);
}

vec3 adjustSaturation(vec3 color, float saturation) {
    float luminance = dot(color, vec3(0.2126, 0.7152, 0.0722));
    return mix(vec3(luminance), color, saturation);
}

vec3 adjustGamma(vec3 color, float gamma) {
    return pow(color, vec3(1.0 / gamma));
}

vec3 applyVignette(vec3 color, vec2 texCoord) {
    vec2 center = vec2(0.5, 0.5);
    float dist = length(texCoord - center);
    float vignette = smoothstep(0.7, 0.2, dist);
    return color * vignette;
}

vec3 applyBloom(vec3 color, vec2 texCoord, sampler2D tex) {
    // MAXIMUM bloom effect for extreme visibility
    float brightness = dot(color, vec3(0.2126, 0.7152, 0.0722));
    
    // Apply bloom to ALL non-black pixels
    if (brightness > 0.05) {  // Was 0.3
        vec2 texSize = textureSize(tex, 0);
        vec2 texelSize = 1.0 / texSize;
        float totalWeight = 0.0;
        vec3 bloomColor = vec3(0.0);
        
        // ULTRA-WIDE 9x9 blur for dramatic bloom effect
        for (int x = -4; x <= 4; x++) {
            for (int y = -4; y <= 4; y++) {
                vec2 offset = vec2(float(x), float(y)) * texelSize;
                float weight = 1.0 / (1.0 + float(abs(x) + abs(y)));
                totalWeight += weight;
                bloomColor += texture(tex, texCoord + offset).rgb * weight;
            }
        }
        
        bloomColor /= totalWeight;
        
        // MAXIMUM bloom effect - apply to nearly all pixels
        float bloomFactor = smoothstep(0.05, 0.5, brightness) * bloomStrength;
        
        // ULTRA-INTENSE bloom mixing
        return mix(color, max(color, bloomColor * 2.0), bloomFactor);
    }
    
    return color;
}

void main()
{
    // Sample the texture
    vec3 color = texture(screenTexture, TexCoords).rgb;
    
    // Apply MAXIMUM bloom effect
    color = applyBloom(color, TexCoords, screenTexture);
    
    // Apply MAXIMUM exposure
    color = adjustExposure(color, exposure);
    
    // Apply MAXIMUM contrast
    color = adjustContrast(color, contrast);
    
    // Apply MAXIMUM saturation
    color = adjustSaturation(color, saturation);
    
    // Apply vignette effect if enabled
    if (vignette) {
        color = applyVignette(color, TexCoords);
    }
    
    // Apply gamma correction
    color = adjustGamma(color, gamma);
    
    // CLAMP to avoid rendering issues with excessive brightness
    color = clamp(color, 0.0, 1.0);
    
    // Set the output color
    FragColor = vec4(color, 1.0);
}