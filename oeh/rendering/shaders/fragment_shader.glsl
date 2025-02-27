#version 330 core
out vec4 FragColor;
in vec2 TexCoords;

uniform sampler2D screenTexture;

// Post-processing parameters
uniform float exposure;       // Controls overall brightness
uniform float contrast;       // Enhances color separation
uniform float saturation;     // Color vibrance
uniform float gamma;          // Gamma correction value
uniform float bloomStrength;  // Intensity of glow around bright areas
uniform bool enableVignette;  // Darkens edges for dramatic effect

// Physics-based tone mapping inspired by Pariev et al. (2003)
vec3 adjustExposure(vec3 color, float exposure) {
    // HDR tone mapping with physical model - simulates camera exposure
    return vec3(1.0) - exp(-color * exposure);
}

vec3 adjustContrast(vec3 color, float contrast) {
    // Improved contrast adjustment with proper luminance preservation
    const vec3 luminanceWeights = vec3(0.2126, 0.7152, 0.0722);
    float luminance = dot(color, luminanceWeights);
    return mix(vec3(luminance), color, contrast);
}

vec3 adjustSaturation(vec3 color, float saturation) {
    // Saturation adjustment with proper luminance preservation
    const vec3 luminanceWeights = vec3(0.2126, 0.7152, 0.0722);
    float luminance = dot(color, luminanceWeights);
    return mix(vec3(luminance), color, saturation);
}

vec3 adjustGamma(vec3 color, float gamma) {
    // Gamma correction for proper color space transformation
    return pow(max(color, vec3(0.0001)), vec3(1.0 / gamma));
}

vec3 applyVignette(vec3 color, vec2 texCoord) {
    // Improved smooth vignette effect
    vec2 center = vec2(0.5);
    float dist = length(texCoord - center);
    float radius = 1.3;  // Adjust for vignette size
    float softness = 0.8; // Adjust for vignette softness
    float vignette = smoothstep(radius, radius - softness, dist);
    return color * vignette;
}

vec3 applyBloom(vec3 color, vec2 texCoord, float strength) {
    // Advanced bloom effect using multi-pass sampling
    const vec3 luminanceWeights = vec3(0.2126, 0.7152, 0.0722);
    float brightness = dot(color, luminanceWeights);
    
    // Only apply bloom to bright areas
    if (brightness > 0.7) {
        vec2 texSize = textureSize(screenTexture, 0);
        vec2 texelSize = 1.0 / texSize;
        
        // Multi-sample blur
        vec3 bloomColor = vec3(0.0);
        float totalWeight = 0.0;
        
        // Two-pass Gaussian approximation
        for (int x = -3; x <= 3; x++) {
            for (int y = -3; y <= 3; y++) {
                // Gaussian weight
                float weight = exp(-(x*x + y*y) / 8.0);
                vec2 offset = vec2(float(x), float(y)) * texelSize * 2.0;
                bloomColor += texture(screenTexture, texCoord + offset).rgb * weight;
                totalWeight += weight;
            }
        }
        
        bloomColor /= totalWeight;
        
        // Adaptive bloom factor based on brightness
        float bloomFactor = smoothstep(0.7, 1.0, brightness) * strength;
        
        // Add bloom to original color
        return color + bloomColor * bloomFactor;
    }
    
    return color;
}

void main() {
    // Sample the texture
    vec3 color = texture(screenTexture, TexCoords).rgb;
    
    // Apply post-processing pipeline
    color = applyBloom(color, TexCoords, bloomStrength);
    color = adjustExposure(color, exposure);
    color = adjustContrast(color, contrast);
    color = adjustSaturation(color, saturation);
    if (enableVignette) {
        color = applyVignette(color, TexCoords);
    }
    color = adjustGamma(color, gamma);
    
    // Final color output
    FragColor = vec4(color, 1.0);
}