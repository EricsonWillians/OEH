#version 330 core
out vec4 FragColor;

in vec2 TexCoords;

uniform sampler2D screenTexture;

// Post-processing parameters
const float exposure = 1.5;        // Increases overall brightness
const float contrast = 1.2;        // Enhances color contrast
const float saturation = 1.4;      // Makes colors more vivid
const float gamma = 2.2;           // Standard gamma correction
const float bloomStrength = 0.3;   // Strength of glow effect around bright areas
const bool vignette = true;        // Darkens the edges for dramatic effect

vec3 adjustExposure(vec3 color, float exposure) {
    return vec3(1.0) - exp(-color * exposure);
}

vec3 adjustContrast(vec3 color, float contrast) {
    return 0.5 + contrast * (color - 0.5);
}

vec3 adjustSaturation(vec3 color, float saturation) {
    // Convert to grayscale
    float luminance = dot(color, vec3(0.2126, 0.7152, 0.0722));
    // Mix between grayscale and original color based on saturation
    return mix(vec3(luminance), color, saturation);
}

vec3 adjustGamma(vec3 color, float gamma) {
    return pow(color, vec3(1.0 / gamma));
}

vec3 applyVignette(vec3 color, vec2 texCoord) {
    // Calculate distance from center
    vec2 center = vec2(0.5, 0.5);
    float dist = length(texCoord - center);
    
    // Create a vignette effect
    float vignette = smoothstep(0.8, 0.2, dist);
    
    return color * vignette;
}

vec3 applyBloom(vec3 color, vec2 texCoord, sampler2D tex) {
    // Simple bloom effect to make bright areas glow
    // This creates a dramatic effect for the accretion disk and stars
    
    // Find bright areas
    float brightness = dot(color, vec3(0.2126, 0.7152, 0.0722));
    
    // Apply bloom only to bright areas
    if (brightness > 0.5) {
        // Sample surrounding pixels
        vec2 texSize = textureSize(tex, 0);
        vec2 texelSize = 1.0 / texSize;
        float totalWeight = 0.0;
        vec3 bloomColor = vec3(0.0);
        
        // Simple 5x5 blur for bloom effect
        for (int x = -2; x <= 2; x++) {
            for (int y = -2; y <= 2; y++) {
                vec2 offset = vec2(float(x), float(y)) * texelSize;
                // Weight decreases with distance from center
                float weight = 1.0 / (1.0 + float(abs(x) + abs(y)));
                totalWeight += weight;
                bloomColor += texture(tex, texCoord + offset).rgb * weight;
            }
        }
        
        // Normalize
        bloomColor /= totalWeight;
        
        // Only use bloom for bright areas and intensify their glow
        float bloomFactor = smoothstep(0.5, 1.0, brightness) * bloomStrength;
        return mix(color, max(color, bloomColor), bloomFactor);
    }
    
    return color;
}

void main()
{
    // Sample the texture
    vec3 color = texture(screenTexture, TexCoords).rgb;
    
    // Apply bloom effect for dramatic glow
    color = applyBloom(color, TexCoords, screenTexture);
    
    // Apply HDR tone mapping with exposure
    color = adjustExposure(color, exposure);
    
    // Apply contrast
    color = adjustContrast(color, contrast);
    
    // Apply saturation
    color = adjustSaturation(color, saturation);
    
    // Apply vignette effect if enabled
    if (vignette) {
        color = applyVignette(color, TexCoords);
    }
    
    // Apply gamma correction
    color = adjustGamma(color, gamma);
    
    // Set the output color
    FragColor = vec4(color, 1.0);
}