#version 330 core
out vec4 FragColor;
in vec2 TexCoords;

uniform sampler2D screenTexture;

// Post-processing parameters
uniform float exposure;       // Overall brightness multiplier
uniform float contrast;       // Contrast adjustment factor
uniform float saturation;     // Saturation adjustment factor
uniform float gamma;          // Gamma correction (final power-law)
uniform float bloomStrength;  // Bloom intensity
uniform bool  enableVignette; // Toggle vignette effect
uniform float boost;          // Extra boost for very low radiance
uniform float time;           // Time for subtle animation effects

// -----------------------------------------------------------------------------
// IMPROVED BLACK HOLE PALETTE AND EFFECTS
// -----------------------------------------------------------------------------
vec3 realisticBlackHolePalette(vec3 color) {
    // Transform original colors to realistic black hole palette
    // Accretion disk: bright white-blue inner region, yellow-orange-red outer regions
    float luminance = dot(color, vec3(0.2126, 0.7152, 0.0722));
    
    // Deep black for shadow/event horizon
    vec3 shadowColor = vec3(0.0, 0.0, 0.0);
    // Intense white-blue for inner disk (extremely hot plasma)
    vec3 innerDiskColor = vec3(0.97, 0.98, 1.0);
    // Blue-white transition for very hot regions
    vec3 hotDiskColor = vec3(0.8, 0.9, 1.0);
    // Yellow-white for mid disk
    vec3 midDiskColor = vec3(1.0, 0.9, 0.7);
    // Orange-red for outer disk
    vec3 outerDiskColor = vec3(1.0, 0.4, 0.1);
    
    // Mix between these colors based on luminance - more color bands for realism
    vec3 result;
    if (luminance < 0.05) {
        // Shadow to inner transition (black to bright)
        float t = luminance / 0.05;
        result = mix(shadowColor, innerDiskColor, t);
    } else if (luminance < 0.2) {
        // Inner disk to hot disk
        float t = (luminance - 0.05) / 0.15;
        result = mix(innerDiskColor, hotDiskColor, t);
    } else if (luminance < 0.5) {
        // Hot disk to mid disk
        float t = (luminance - 0.2) / 0.3;
        result = mix(hotDiskColor, midDiskColor, t);
    } else {
        // Mid to outer disk
        float t = (luminance - 0.5) / 0.5;
        result = mix(midDiskColor, outerDiskColor, t);
    }
    
    return result;
}

// -----------------------------------------------------------------------------
// ENHANCED TONE MAPPING WITH MORE DRAMATIC CONTRAST
// -----------------------------------------------------------------------------
vec3 dramaticToneMapping(vec3 x)
{
    // Higher contrast, higher dynamic range tone mapping
    float A = 2.8;  // Increased from 2.51
    float B = 0.02; // Decreased from 0.03 for darker blacks
    float C = 2.7;  // Increased from 2.43
    float D = 0.59;
    float E = 0.09; // Decreased from 0.14 for more dramatic contrast
    
    vec3 result = clamp((x * (A * x + B)) / (x * (C * x + D) + E), 0.0, 1.0);
    
    // Apply realistic palette
    return realisticBlackHolePalette(result);
}

// -----------------------------------------------------------------------------
// IMPROVED BLOOM EFFECT FOR ACCRETION DISK GLOW
// -----------------------------------------------------------------------------
vec3 applyAccretionDiskBloom(vec3 color, vec2 uv, float strength)
{
    const vec3 lumW = vec3(0.2126, 0.7152, 0.0722);
    float brightness = dot(color, lumW);
    
    // Apply bloom only to bright areas (accretion disk)
    if (brightness > 0.5) { // Lowered threshold from 0.6 to catch more of the disk
        vec2 texSize = textureSize(screenTexture, 0);
        vec2 texel = 1.0 / texSize;
        vec3 sum = vec3(0.0);
        float total = 0.0;
        
        // Larger blur radius for accretion disk glow
        for (int x = -6; x <= 6; x++) { // Increased from -4/4 to -6/6
            for (int y = -6; y <= 6; y++) {
                float weight = exp(-float(x*x + y*y) / 12.0); // Increased from 8.0 to 12.0
                vec2 offset = vec2(x, y) * texel * 2.5; // Increased from 2.0 to 2.5
                sum += texture(screenTexture, uv + offset).rgb * weight;
                total += weight;
            }
        }
        sum /= total;
        
        // Make bloom more white-blue tinted for inner disk
        sum = mix(sum, vec3(sum.r * 0.9, sum.g * 0.9, sum.b * 1.3), 0.4); // Increased blue tint
        
        // Apply stronger bloom effect to accretion disk
        float bloomFactor = smoothstep(0.5, 1.0, brightness) * strength * 2.0; // Increased from 1.5 to 2.0
        return color + sum * bloomFactor;
    }
    return color;
}

// -----------------------------------------------------------------------------
// ENHANCED CONTRAST ADJUSTMENT
// -----------------------------------------------------------------------------
vec3 adjustContrast(vec3 color, float contrast)
{
    const vec3 lumW = vec3(0.2126, 0.7152, 0.0722);
    float lum = dot(color, lumW);
    
    // Apply stronger contrast to make black hole darker and disk brighter
    return mix(vec3(lum), color, contrast * 1.2); // Increased contrast multiplier
}

// -----------------------------------------------------------------------------
// ENHANCED VIGNETTE EFFECT - SUBTLE DARKENING AT EDGES
// -----------------------------------------------------------------------------
vec3 applyVignette(vec3 color, vec2 uv)
{
    vec2 center = vec2(0.5);
    float dist = length(uv - center);
    
    // More dramatic vignette
    float radius = 1.3; // Decreased from 1.4 for stronger effect
    float softness = 0.7; // Decreased from 0.8 for harder edge
    
    // Compute vignette factor
    float vig = smoothstep(radius, radius - softness, dist);
    
    // Apply vignette
    return color * vig;
}

// -----------------------------------------------------------------------------
// ENHANCED CHROMATIC ABERRATION - MORE PRONOUNCED COLOR SEPARATION AT EDGES
// -----------------------------------------------------------------------------
vec3 applyChromaticAberration(vec2 uv) {
    float aberrationAmount = 0.003; // Increased from 0.0015 for more noticeable effect
    
    // Calculate distortion based on distance from center
    vec2 center = vec2(0.5);
    vec2 dist = uv - center;
    
    // Stronger effect toward edges
    float distAmount = length(dist) * 2.5; // Increased from 2.0
    aberrationAmount *= smoothstep(0.0, 1.0, distAmount);
    
    // Sample colors with more pronounced channel separation
    vec3 result;
    result.r = texture(screenTexture, uv - dist * aberrationAmount * 1.2).r; // 20% more red separation
    result.g = texture(screenTexture, uv).g;
    result.b = texture(screenTexture, uv + dist * aberrationAmount * 1.5).b; // 50% more blue separation
    
    return result;
}

// -----------------------------------------------------------------------------
// ENHANCED SPACE BACKGROUND WITH MORE STARS AND NEBULAE
// -----------------------------------------------------------------------------
vec3 createSpaceBackground(vec2 uv) {
    // Create a deeper space background with more stars
    vec3 backgroundColor = vec3(0.008, 0.01, 0.04); // Darker background
    
    // Star pattern with more variation
    float starPattern = fract(sin(dot(uv, vec2(12.9898, 78.233))) * 43758.5453);
    float starPattern2 = fract(sin(dot(uv * 1.5, vec2(63.1237, 21.456))) * 93156.4872); // Additional pattern
    
    // Create stars of different sizes and brightness - more stars
    if (starPattern > 0.998) {
        // Very bright star
        return vec3(1.0, 1.0, 1.0);
    } else if (starPattern > 0.995) {
        // Medium bright blue-white star
        return vec3(0.8, 0.9, 1.0);
    } else if (starPattern > 0.991) { // Increased from 0.992
        // Yellow star
        return vec3(1.0, 0.9, 0.6);
    } else if (starPattern > 0.986) { // Increased from 0.989
        // Faint red star
        return vec3(0.8, 0.4, 0.3);
    } else if (starPattern2 > 0.994) {
        // Additional blue stars
        return vec3(0.5, 0.7, 1.0);
    }
    
    // Add enhanced nebula effects
    float nebulaPattern = fract(sin(dot(uv * 0.5, vec2(8.7543, 43.1684))) * 32768.5453);
    float nebulaPattern2 = fract(sin(dot(uv * 0.3, vec2(18.423, 84.321))) * 18532.7621); // Additional pattern
    
    if (nebulaPattern > 0.72) { // Increased from 0.75
        float intensity = (nebulaPattern - 0.72) * 0.1; // Increased from 0.08
        vec3 nebulaColor;
        
        // Different nebula colors based on position
        if (uv.x + uv.y > 1.0) {
            nebulaColor = vec3(0.07, 0.01, 0.1) * intensity; // More purple
        } else {
            nebulaColor = vec3(0.01, 0.05, 0.08) * intensity; // More blue
        }
        
        backgroundColor += nebulaColor;
    }
    
    if (nebulaPattern2 > 0.8) {
        float intensity = (nebulaPattern2 - 0.8) * 0.08;
        vec3 nebulaColor = vec3(0.08, 0.02, 0.0) * intensity; // Reddish nebula
        backgroundColor += nebulaColor;
    }
    
    return backgroundColor;
}

// -----------------------------------------------------------------------------
// ENHANCED GRAVITATIONAL LENSING EFFECT
// -----------------------------------------------------------------------------
vec3 simulateGravitationalLensing(vec2 uv, vec3 diskColor) {
    // Simulate gravitational lensing around the black hole
    vec2 center = vec2(0.5, 0.5);
    float dist = length(uv - center);
    
    // Photon sphere and event horizon radii (in normalized coordinates)
    float eventHorizonRadius = 0.05;
    float photonSphereRadius = 0.08;
    float lensingSphereRadius = 0.28; // Increased from 0.2 for wider effect
    
    // Distance-based effects
    if (dist < eventHorizonRadius) {
        // Inside event horizon - completely black
        return vec3(0.0);
    } else if (dist < photonSphereRadius) {
        // Between event horizon and photon sphere - strong lensing
        float lensStrength = (dist - eventHorizonRadius) / (photonSphereRadius - eventHorizonRadius);
        
        // Create the bright photon ring at the edge of the black hole (Einstein ring)
        if (dist > 0.068 && dist < 0.078) { // Wider ring
            return mix(vec3(0.0), vec3(1.0, 0.95, 0.9), pow(lensStrength, 0.2)); // Brighter ring
        } else {
            return mix(vec3(0.0), diskColor * 0.3, pow(lensStrength, 2.0)); // Darker inside
        }
    } else if (dist < lensingSphereRadius) {
        // Wider lensing area around black hole
        float lensStrength = (dist - photonSphereRadius) / (lensingSphereRadius - photonSphereRadius);
        
        // Calculate distortion direction and amount
        vec2 direction = normalize(uv - center);
        float distortionAmount = 0.08 * (1.0 - lensStrength); // Increased from 0.05
        
        // Add time-based subtle pulsation to the distortion
        distortionAmount *= 1.0 + 0.05 * sin(time * 0.3); // Subtle pulsation effect
        
        // Sample the texture with distortion to simulate lensing
        vec2 distortedUV = uv - direction * distortionAmount;
        vec3 sampledColor = texture(screenTexture, distortedUV).rgb;
        
        // Enhanced brightness around the edge due to light concentration
        if (dist < 0.12) { // Increased from 0.1
            sampledColor *= 1.8; // Increased from 1.5
        }
        
        return sampledColor;
    }
    
    // Outside strong lensing area - minimal effects
    return diskColor;
}

// -----------------------------------------------------------------------------
// ENHANCED ACCRETION DISK RENDERING
// -----------------------------------------------------------------------------
vec3 renderAccretionDisk(vec2 uv) {
    vec2 center = vec2(0.5, 0.5);
    float dist = length(uv - center);
    
    // Accretion disk geometry
    float innerRadius = 0.09;  // Just outside photon sphere
    float outerRadius = 0.45;  // Increased from 0.4 for wider disk
    
    if (dist > innerRadius && dist < outerRadius) {
        // Calculate point position in polar coordinates
        float angle = atan(uv.y - center.y, uv.x - center.x);
        
        // Calculate temperature based on distance from black hole
        // Temperature decreases with distance (T ∝ r^(-3/4) in standard model)
        float temp = 1.0 - pow((dist - innerRadius) / (outerRadius - innerRadius), 0.75);
        
        // Add more pronounced spiral structure to the disk
        float spiral = sin(angle * 5.0 + 18.0 * (dist - innerRadius) + time * 0.3) * 0.6 + 0.5; // More spirals, faster movement
        temp = mix(temp, temp * spiral, 0.4); // Increased spiral influence from 0.3 to 0.4
        
        // Add turbulence/non-uniformity to the disk
        float turbulence = fract(sin(dot(uv * 12.0, vec2(12.9898, 78.233))) * 43758.5453); // Increased from 10.0 to 12.0
        float turbulence2 = fract(sin(dot(uv * 8.0 + time * 0.1, vec2(26.651, 36.477))) * 85429.1234); // Time-based turbulence
        turbulence = mix(turbulence, turbulence2, 0.3); // Mix in time-based turbulence
        temp = mix(temp, temp * turbulence, 0.2); // Increased from 0.15 to 0.2
        
        // Generate disk color based on temperature with more dramatic coloration
        vec3 diskColor;
        if (temp > 0.8) {
            // Hottest inner region - blue-white, more intense
            diskColor = mix(vec3(0.8, 0.9, 1.0), vec3(1.0, 1.0, 1.0), (temp - 0.8) * 5.0);
            // Add slight emission glow for hottest parts
            diskColor *= 1.2;
        } else if (temp > 0.5) {
            // Hot region - white-yellow
            diskColor = mix(vec3(1.0, 0.8, 0.4), vec3(0.8, 0.9, 1.0), (temp - 0.5) * 3.33);
        } else if (temp > 0.3) {
            // Medium region - orange
            diskColor = mix(vec3(0.9, 0.4, 0.1), vec3(1.0, 0.8, 0.4), (temp - 0.3) * 5.0);
        } else {
            // Outer region - redder
            diskColor = mix(vec3(0.5, 0.1, 0.03), vec3(0.9, 0.4, 0.1), temp * 3.33);
        }
        
        // Fade out disk at outer edge with slight glow effect
        float edgeFade = smoothstep(outerRadius, outerRadius - 0.1, dist);
        diskColor *= edgeFade;
        
        // Fade out disk at inner edge (consumption by black hole)
        float innerFade = smoothstep(innerRadius, innerRadius + 0.02, dist);
        diskColor *= innerFade;
        
        // Apply disk thickness effect (thinner when viewed edge-on)
        // Here we assume disk is somewhat in the equatorial plane
        float diskFalloff = 1.0 - abs(uv.y - center.y) * 5.0; // Make the disk more concentrated in the center
        diskFalloff = max(0.0, diskFalloff);
        
        // Time-based disk wobble - subtle disk precession effect
        float wobble = sin(time * 0.2) * 0.01; // Subtle wobble amount
        diskFalloff = 1.0 - abs((uv.y - center.y) + wobble) * 5.0;
        diskFalloff = max(0.0, diskFalloff);
        
        diskColor *= pow(diskFalloff, 1.8); // Decreased from 2.0 for slightly thicker appearance
        
        return diskColor;
    }
    
    return vec3(0.0);
}

// -----------------------------------------------------------------------------
// LIGHT STREAKING EFFECT - SIMULATING LIGHT BEING SUCKED IN
// -----------------------------------------------------------------------------
vec3 applyLightStreaking(vec3 color, vec2 uv) {
    vec2 center = vec2(0.5, 0.5);
    vec2 dir = normalize(center - uv);
    float dist = length(uv - center);
    
    // Only apply streaking in the area around the black hole
    if (dist > 0.08 && dist < 0.3) {
        float streakIntensity = smoothstep(0.3, 0.08, dist) * 0.7; // Stronger closer to the black hole
        
        // Add time-based variation
        streakIntensity *= 1.0 + 0.1 * sin(time * 0.5 + dist * 10.0);
        
        // Sample points along the direction vector
        vec3 streak = vec3(0.0);
        float totalWeight = 0.0;
        
        for (int i = 1; i <= 10; i++) {
            float weight = float(11 - i) / 10.0 * streakIntensity;
            vec2 offset = dir * float(i) * 0.01 * streakIntensity;
            streak += texture(screenTexture, uv + offset).rgb * weight;
            totalWeight += weight;
        }
        
        if (totalWeight > 0.0) {
            streak /= totalWeight;
            // Blend the streak effect with the original color
            return mix(color, streak, streakIntensity * 0.4);
        }
    }
    
    return color;
}

// -----------------------------------------------------------------------------
// MAIN - COMPLETELY REWRITTEN FOR REALISTIC BLACK HOLE
// -----------------------------------------------------------------------------
void main()
{
    // Create space background with stars
    vec3 background = createSpaceBackground(TexCoords);
    
    // Render the accretion disk
    vec3 diskColor = renderAccretionDisk(TexCoords);
    
    // Apply gravitational lensing
    vec3 lensedColor = simulateGravitationalLensing(TexCoords, diskColor + background);
    
    // Combine disk and background
    vec3 color = lensedColor + diskColor;
    
    // Apply light streaking effect (being sucked into black hole)
    color = applyLightStreaking(color, TexCoords);
    
    // Apply chromatic aberration for more dramatic edge distortion
    color = mix(color, applyChromaticAberration(TexCoords), 0.3);
    
    // Apply bloom effect for disk glow
    color = applyAccretionDiskBloom(color, TexCoords, bloomStrength * 3.0); // Increased from 2.0 to 3.0
    
    // Apply exposure adjustment
    color *= exposure;
    
    // Apply saturation adjustment with boost for more vivid colors
    float saturationBoost = saturation * 1.2; // Add 20% more saturation
    color = mix(vec3(dot(color, vec3(0.2126, 0.7152, 0.0722))), color, saturationBoost);
    
    // Apply contrast and tone mapping
    color = adjustContrast(color, contrast);
    color = dramaticToneMapping(color);
    
    // Apply vignette if enabled
    if (enableVignette) {
        color = applyVignette(color, TexCoords);
    }
    
    // Final gamma correction
    color = pow(max(color, vec3(0.0)), vec3(1.0 / gamma));
    
    // Ensure event horizon is completely black with enhanced edge glow
    vec2 center = vec2(0.5, 0.5);
    float distToCenter = length(TexCoords - center);
    if (distToCenter < 0.05) {
        // Create perfect black circle for event horizon with enhanced edge glow
        float horizonFade = smoothstep(0.045, 0.05, distToCenter);
        
        // Add more intense blue-white glow at the very edge of the event horizon
        // This creates the dramatic "last photon orbit" effect
        vec3 edgeGlow = vec3(0.5, 0.8, 1.0) * pow(horizonFade, 6.0) * 3.0; // Brighter glow
        
        // Add subtle time-based pulsation to the edge glow
        edgeGlow *= 1.0 + 0.2 * sin(time * 0.8);
        
        // Mix the black hole with the scene
        color = mix(vec3(0.0) + edgeGlow, color, horizonFade);
    }
    
    // Ensure final color is properly clamped
    color = clamp(color, 0.0, 1.0);
    
    FragColor = vec4(color, 1.0);
}