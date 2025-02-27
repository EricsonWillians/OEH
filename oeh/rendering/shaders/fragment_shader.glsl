#version 330 core
out vec4 FragColor;
in vec2 TexCoords;

uniform sampler2D screenTexture;

// Post‑processing parameters
uniform float exposure;       // Overall brightness multiplier
uniform float contrast;       // Contrast adjustment factor
uniform float saturation;     // Saturation adjustment factor
uniform float gamma;          // Gamma correction (final power-law)
uniform float bloomStrength;  // Bloom intensity
uniform bool  enableVignette; // Toggle vignette effect
uniform float boost;          // Extra boost for very low radiance

// -----------------------------------------------------------------------------
// FILMIC TONE MAPPING (ACES‑inspired)
// -----------------------------------------------------------------------------
vec3 filmicToneMapping(vec3 x)
{
    // Lift shadows a bit so nothing stays pitch black.
    x = max(vec3(0.0), x - 0.004);
    
    // ACES tone mapping constants (tweak as needed)
    float A = 2.51;
    float B = 0.03;
    float C = 2.43;
    float D = 0.59;
    float E = 0.14;
    
    return clamp((x * (A * x + B)) / (x * (C * x + D) + E), 0.0, 1.0);
}

// -----------------------------------------------------------------------------
// BLOOM: Gaussian blur on bright areas
// -----------------------------------------------------------------------------
vec3 applyBloom(vec3 color, vec2 uv, float strength)
{
    const vec3 lumW = vec3(0.2126, 0.7152, 0.0722);
    float brightness = dot(color, lumW);
    if (brightness > 0.7) {
        vec2 texSize = textureSize(screenTexture, 0);
        vec2 texel = 1.0 / texSize;
        vec3 sum = vec3(0.0);
        float total = 0.0;
        for (int x = -3; x <= 3; x++) {
            for (int y = -3; y <= 3; y++) {
                float weight = exp(-float(x*x + y*y) / 8.0);
                vec2 offset = vec2(x, y) * texel * 2.0;
                sum += texture(screenTexture, uv + offset).rgb * weight;
                total += weight;
            }
        }
        sum /= total;
        float bloomFactor = smoothstep(0.7, 1.0, brightness) * strength;
        return color + sum * bloomFactor;
    }
    return color;
}

// -----------------------------------------------------------------------------
// CONTRAST ADJUSTMENT
// -----------------------------------------------------------------------------
vec3 adjustContrast(vec3 color, float contrast)
{
    const vec3 lumW = vec3(0.2126, 0.7152, 0.0722);
    float lum = dot(color, lumW);
    return mix(vec3(lum), color, contrast);
}

// -----------------------------------------------------------------------------
// SATURATION ADJUSTMENT
// -----------------------------------------------------------------------------
vec3 adjustSaturation(vec3 color, float saturation)
{
    const vec3 lumW = vec3(0.2126, 0.7152, 0.0722);
    float lum = dot(color, lumW);
    return mix(vec3(lum), color, saturation);
}

// -----------------------------------------------------------------------------
// VIGNETTE EFFECT
// -----------------------------------------------------------------------------
vec3 applyVignette(vec3 color, vec2 uv)
{
    vec2 center = vec2(0.5);
    float dist = length(uv - center);
    float radius = 1.3;
    float softness = 0.8;
    float vig = smoothstep(radius, radius - softness, dist);
    return color * vig;
}

// -----------------------------------------------------------------------------
// MAIN
// -----------------------------------------------------------------------------
void main()
{
    // Sample the texture from the raytracer.
    vec3 color = texture(screenTexture, TexCoords).rgb;
    
    // If the sampled color is extremely dark, boost it.
    float maxChannel = max(color.r, max(color.g, color.b));
    if(maxChannel < 0.01) {
        color *= boost;
    }
    
    // Apply exposure scaling.
    color *= exposure;
    
    // Apply bloom effect.
    color = applyBloom(color, TexCoords, bloomStrength);
    
    // Adjust contrast and saturation.
    color = adjustContrast(color, contrast);
    color = adjustSaturation(color, saturation);
    
    // Apply filmic tone mapping.
    color = filmicToneMapping(color);
    
    // Optional vignette effect.
    if(enableVignette) {
        color = applyVignette(color, TexCoords);
    }
    
    // Final gamma correction.
    color = pow(max(color, vec3(0.0)), vec3(1.0 / gamma));
    
    // Clamp final color.
    color = clamp(color, 0.0, 1.0);
    
    FragColor = vec4(color, 1.0);
}
