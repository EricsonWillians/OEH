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

vec3 adjustContrast(vec3 color, float contrast)
{
    const vec3 lumW = vec3(0.2126, 0.7152, 0.0722);
    float lum = dot(color, lumW);
    
    // Apply stronger contrast to make black hole darker and disk brighter
    return mix(vec3(lum), color, contrast * 1.3); // Increased contrast multiplier
}

// -----------------------------------------------------------------------------
// DARK VOID BLACK HOLE PALETTE - FOCUS ON DARKNESS AND CONTRAST
// -----------------------------------------------------------------------------
vec3 darkVoidPalette(vec3 color) {
    // Transform original colors to emphasize darkness with occasional bright highlights
    float luminance = dot(color, vec3(0.2126, 0.7152, 0.0722));
    
    // Absolute black for shadow/event horizon
    vec3 shadowColor = vec3(0.0, 0.0, 0.0);
    // Cold blue-white for inner disk (extremely hot plasma)
    vec3 innerDiskColor = vec3(0.8, 0.95, 1.0);
    // Electric blue for hot regions
    vec3 hotDiskColor = vec3(0.3, 0.7, 1.0);
    // Faint cyan-green for mid disk
    vec3 midDiskColor = vec3(0.2, 0.5, 0.4);
    // Very dark blue for outer disk
    vec3 outerDiskColor = vec3(0.05, 0.07, 0.15);
    
    // Mix between these colors based on luminance
    vec3 result;
    if (luminance < 0.05) {
        // Shadow to inner transition (black to bright)
        float t = luminance / 0.05;
        result = mix(shadowColor, innerDiskColor, pow(t, 0.8)); // Sharper transition
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
        // Force darker outer regions
        result *= 0.7;
    }
    
    return result;
}

// -----------------------------------------------------------------------------
// EXTREME CONTRAST TONE MAPPING - EMPHASIZE THE VOID
// -----------------------------------------------------------------------------
vec3 extremeContrastToneMapping(vec3 x)
{
    // Very high contrast, emphasizing the black void
    float A = 3.5;   // Increased dramatically
    float B = 0.005; // Nearly zero for deep blacks
    float C = 3.8;   // Increased for extreme contrast
    float D = 0.5;
    float E = 0.03;  // Very small for extreme contrast
    
    vec3 result = clamp((x * (A * x + B)) / (x * (C * x + D) + E), 0.0, 1.0);
    
    // Apply dark void palette
    return darkVoidPalette(result);
}

// -----------------------------------------------------------------------------
// ENHANCED BLOOM EFFECT FOR COLD, EERIE GLOW
// -----------------------------------------------------------------------------
vec3 applyEerieBloom(vec3 color, vec2 uv, float strength)
{
    const vec3 lumW = vec3(0.2126, 0.7152, 0.0722);
    float brightness = dot(color, lumW);
    
    // Apply bloom only to brightest areas
    if (brightness > 0.6) { // Higher threshold for more focused, intense bloom
        vec2 texSize = textureSize(screenTexture, 0);
        vec2 texel = 1.0 / texSize;
        vec3 sum = vec3(0.0);
        float total = 0.0;
        
        // Wider blur radius for atmospheric disk glow
        for (int x = -8; x <= 8; x++) {
            for (int y = -8; y <= 8; y++) {
                float weight = exp(-float(x*x + y*y) / 24.0); // Wider spread
                vec2 offset = vec2(x, y) * texel * 3.0;
                sum += texture(screenTexture, uv + offset).rgb * weight;
                total += weight;
            }
        }
        sum /= total;
        
        // Make bloom cold blue-white for an eerie glow
        sum = mix(sum, vec3(sum.b * 0.9, sum.g * 0.8, sum.b * 1.3), 0.7);
        
        // Apply stronger bloom effect to brightest regions only
        float bloomFactor = pow(smoothstep(0.6, 1.0, brightness), 2.0) * strength * 4.0;
        
        // Subtle pulsation to the bloom intensity based on time
        bloomFactor *= 1.0 + 0.3 * sin(time * 0.3);
        
        return color + sum * bloomFactor;
    }
    return color;
}

// -----------------------------------------------------------------------------
// EXTREME VIGNETTE EFFECT - DARKNESS CLOSING IN
// -----------------------------------------------------------------------------
vec3 applyExtremeDarkVignette(vec3 color, vec2 uv)
{
    vec2 center = vec2(0.5);
    float dist = length(uv - center);
    
    // Stronger, darker vignette
    float radius = 1.0; // Decreased for stronger effect
    float softness = 0.4; // Decreased for harder edge
    
    // Compute vignette factor
    float vig = smoothstep(radius, radius - softness, dist);
    
    // Make the vignette pulsate slightly
    vig *= 0.85 + 0.15 * sin(time * 0.2);
    
    // Apply vignette
    return color * vig;
}

// -----------------------------------------------------------------------------
// DARK SPACE BACKGROUND - NEARLY ABSOLUTE VOID
// -----------------------------------------------------------------------------
vec3 createVoidBackground(vec2 uv) {
    // Create an extremely dark space background
    vec3 backgroundColor = vec3(0.001, 0.002, 0.004); // Near black
    
    // Sparse, distant star pattern
    float starPattern = fract(sin(dot(uv, vec2(12.9898, 78.233))) * 43758.5453);
    
    // Create very few, distant stars
    if (starPattern > 0.9995) {
        // Rare bright star
        return vec3(0.8, 0.9, 1.0) * (0.7 + 0.3 * sin(time * 2.0 + uv.x * 10.0)); // Add subtle flickering
    } else if (starPattern > 0.998) {
        // Faint distant star
        return vec3(0.3, 0.4, 0.6) * (0.6 + 0.4 * sin(time * 1.5 + uv.y * 20.0));
    }
    
    // Barely visible dust and gas
    float dustPattern = fract(sin(dot(uv * 0.5, vec2(8.7543, 43.1684))) * 32768.5453);
    
    if (dustPattern > 0.85) {
        float intensity = (dustPattern - 0.85) * 0.02; // Very faint
        vec3 dustColor = vec3(0.02, 0.03, 0.06) * intensity;
        backgroundColor += dustColor;
    }
    
    return backgroundColor;
}

// -----------------------------------------------------------------------------
// ENHANCED GRAVITATIONAL LENSING - MORE EXTREME DISTORTION
// -----------------------------------------------------------------------------
vec3 simulateExtremeLensing(vec2 uv, vec3 diskColor) {
    // Simulate gravitational lensing around the black hole
    vec2 center = vec2(0.5, 0.5);
    float dist = length(uv - center);
    
    // Photon sphere and event horizon radii
    float eventHorizonRadius = 0.07; // Larger event horizon
    float photonSphereRadius = 0.09;
    float lensingSphereRadius = 0.4; // Wider effect
    
    // Distance-based effects
    if (dist < eventHorizonRadius) {
        // Inside event horizon - absolute void
        return vec3(0.0);
    } else if (dist < photonSphereRadius) {
        // Between event horizon and photon sphere - extreme lensing
        float lensStrength = (dist - eventHorizonRadius) / (photonSphereRadius - eventHorizonRadius);
        
        // Create a thin, intense Einstein ring (photon sphere)
        if (dist > eventHorizonRadius + 0.005 && dist < eventHorizonRadius + 0.015) {
            // Make Einstein ring flicker and pulse
            float ringPulse = 0.8 + 0.4 * sin(time * 1.0 + dist * 50.0);
            vec3 ringColor = vec3(0.2, 0.7, 1.0) * ringPulse * 2.0; // Intense blue glow
            return mix(vec3(0.0), ringColor, pow(lensStrength, 0.3));
        } else {
            // Deep darkness near the event horizon
            return vec3(0.0);
        }
    } else if (dist < lensingSphereRadius) {
        // Wider lensing area around black hole
        float lensStrength = (dist - photonSphereRadius) / (lensingSphereRadius - photonSphereRadius);
        
        // Calculate distortion direction and amount
        vec2 direction = normalize(uv - center);
        float distortionAmount = 0.15 * pow((1.0 - lensStrength), 2.0); // More dramatic near center
        
        // Add time-based fluctuation to the distortion
        distortionAmount *= 1.0 + 0.2 * sin(time * 0.15 + dist * 10.0);
        
        // Sample the texture with distortion to simulate extreme lensing
        vec2 distortedUV = uv - direction * distortionAmount;
        vec3 sampledColor = texture(screenTexture, distortedUV).rgb;
        
        // Brightness boost only at certain rings to create eerie patterns
        float ringEffect = sin(dist * 40.0 - time * 0.2) * 0.5 + 0.5;
        float brightnessBoost = smoothstep(0.4, 0.8, ringEffect) * (1.0 - lensStrength) * 0.7;
        
        return sampledColor * (1.0 + brightnessBoost);
    }
    
    return diskColor;
}

// -----------------------------------------------------------------------------
// ENHANCED ACCRETION DISK - COLDER, MORE ALIEN APPEARANCE
// -----------------------------------------------------------------------------
vec3 renderAlienDisk(vec2 uv) {
    vec2 center = vec2(0.5, 0.5);
    float dist = length(uv - center);
    
    // Accretion disk geometry
    float innerRadius = 0.1;
    float outerRadius = 0.6; // Larger disk
    
    if (dist > innerRadius && dist < outerRadius) {
        // Calculate point position in polar coordinates
        float angle = atan(uv.y - center.y, uv.x - center.x);
        
        // Calculate temperature based on distance from black hole
        float temp = 1.0 - pow((dist - innerRadius) / (outerRadius - innerRadius), 0.7);
        
        // Add spiral structure to the disk with sharper features
        float spiralCount = 3.0; // Fewer, more dramatic spiral arms
        float spiralTightness = 35.0; // Tighter spirals
        float spiralStrength = 0.85; // Stronger spiral pattern
        
        // Create spiral patterns with sharp edges
        float spiral = smoothstep(0.45, 0.55, sin(angle * spiralCount + spiralTightness * (dist - innerRadius) + time * 0.15) * 0.5 + 0.5);
        temp = mix(temp * 0.4, temp * 1.1, spiral); // Create dramatic bright/dark contrast in spirals
        
        // Add turbulence/non-uniformity to the disk
        float turbulence = fract(sin(dot(uv * 20.0, vec2(12.9898, 78.233))) * 43758.5453);
        float turbulence2 = fract(sin(dot(uv * 15.0 + time * 0.1, vec2(26.651, 36.477))) * 85429.1234);
        turbulence = mix(turbulence, turbulence2, 0.5);
        
        // Apply turbulence as a multiplier for more dramatic contrast
        temp *= 0.7 + 0.6 * smoothstep(0.4, 0.6, turbulence);
        
        // Generate disk color based on temperature - emphasize cold blues
        vec3 diskColor;
        if (temp > 0.75) {
            // Hottest inner region - bright electric blue
            diskColor = mix(vec3(0.1, 0.5, 1.0), vec3(0.8, 0.95, 1.0), (temp - 0.75) * 4.0);
            diskColor *= 2.0; // Extremely bright inner region
        } else if (temp > 0.5) {
            // Hot region - electric blue to cyan
            diskColor = mix(vec3(0.05, 0.3, 0.6), vec3(0.1, 0.5, 1.0), (temp - 0.5) * 4.0);
        } else if (temp > 0.25) {
            // Medium region - deep blue
            diskColor = mix(vec3(0.02, 0.1, 0.25), vec3(0.05, 0.3, 0.6), (temp - 0.25) * 4.0);
        } else {
            // Outer region - near black with hint of blue
            diskColor = mix(vec3(0.005, 0.01, 0.04), vec3(0.02, 0.1, 0.25), temp * 4.0);
        }
        
        // Fade out disk at outer edge
        float edgeFade = smoothstep(outerRadius, outerRadius - 0.15, dist);
        diskColor *= edgeFade;
        
        // Fade out disk at inner edge (consumption by black hole)
        float innerFade = smoothstep(innerRadius, innerRadius + 0.02, dist);
        diskColor *= innerFade;
        
        // Apply disk thickness effect (thinner when viewed edge-on)
        float diskFalloff = 1.0 - abs(uv.y - center.y) * 6.0; // Thinner disk
        diskFalloff = max(0.0, diskFalloff);
        
        // Time-based disk wobble/precession for instability effect
        float wobbleAmount = 0.03;
        float wobbleSpeed = 0.1;
        float wobble = wobbleAmount * sin(time * wobbleSpeed);
        diskFalloff = 1.0 - abs((uv.y - center.y) + wobble) * 6.0;
        diskFalloff = max(0.0, diskFalloff);
        
        // Apply sharper disk profile
        diskColor *= pow(diskFalloff, 1.4);
        
        // Create occasional energy surges in the disk
        float surgeTiming = sin(time * 0.3) * sin(time * 0.17) * sin(time * 0.43);
        if (surgeTiming > 0.85) {
            float surgeFactor = (surgeTiming - 0.85) * 6.0;
            float surgePattern = smoothstep(0.45, 0.55, sin(angle * 6.0 + dist * 50.0 + time * 3.0));
            diskColor += vec3(0.0, 0.4, 0.8) * surgePattern * surgeFactor;
        }
        
        return diskColor;
    }
    
    return vec3(0.0);
}

// -----------------------------------------------------------------------------
// LIGHT ABSORPTION STREAKS - DARKNESS BEING PULLED IN
// -----------------------------------------------------------------------------
vec3 applyLightAbsorption(vec3 color, vec2 uv) {
    vec2 center = vec2(0.5, 0.5);
    vec2 dir = normalize(center - uv);
    float dist = length(uv - center);
    
    // Apply in wider area around black hole
    if (dist > 0.1 && dist < 0.7) {
        // Calculate streak intensity - stronger closer to hole
        float streakIntensity = 0.35 * pow(1.0 - (dist - 0.1) / 0.6, 2.0);
        
        // Add time variation
        streakIntensity *= 1.0 + 0.3 * sin(time * 0.13 + dist * 8.0);
        
        // Create dark streaks
        float streakFactor = 0.0;
        
        // Sample points along the direction vector toward the black hole
        for (int i = 1; i <= 12; i++) {
            float weight = float(i) / 12.0; // Stronger effect further along streak
            vec2 offset = dir * float(i) * 0.02;
            vec3 sampledColor = texture(screenTexture, uv + offset).rgb;
            float darkness = 1.0 - dot(sampledColor, vec3(0.2126, 0.7152, 0.0722));
            streakFactor += darkness * weight * streakIntensity;
        }
        
        // Apply darkening along streaks
        return color * (1.0 - streakFactor * 0.7);
    }
    
    return color;
}

// -----------------------------------------------------------------------------
// EVENT HORIZON EFFECT - UNSTABLE VOID WITH ENERGY DISCHARGE
// -----------------------------------------------------------------------------
vec3 createUnstableVoid(vec3 color, vec2 uv) {
    vec2 center = vec2(0.5, 0.5);
    float distToCenter = length(uv - center);
    
    // Base event horizon radius
    float baseRadius = 0.07;
    
    // Create unstable, subtly pulsating event horizon
    float pulseAmount = 0.005 * sin(time * 0.4);
    float currentRadius = baseRadius + pulseAmount;
    
    if (distToCenter < currentRadius) {
        // Inside event horizon - absolute void with occasional glimpses
        float voidNoise = fract(sin(dot(uv * 10.0 + time * 0.1, vec2(12.9898, 78.233))) * 43758.5453);
        
        // Extremely rare "glimpses" into the void
        if (voidNoise > 0.997) {
            // Momentary flicker of energy from within
            return vec3(0.1, 0.3, 0.7) * (voidNoise - 0.997) * 300.0;
        }
        
        return vec3(0.0); // Complete blackness
    } else if (distToCenter < currentRadius + 0.015) {
        // Edge effect - thin blue energy discharge at event horizon
        float edgeFactor = 1.0 - (distToCenter - currentRadius) / 0.015;
        
        // Create flickering, unstable edge
        float flicker = 0.6 + 0.4 * sin(time * 10.0 + uv.x * 30.0 + uv.y * 20.0);
        float ringPattern = sin(distToCenter * 200.0 - time * 1.0) * 0.5 + 0.5;
        
        // Create electric blue edge with intensity variation around circumference
        float angle = atan(uv.y - center.y, uv.x - center.x);
        float angularVariation = 0.7 + 0.3 * sin(angle * 8.0 + time * 2.0);
        
        vec3 edgeColor = vec3(0.0, 0.5, 1.0) * pow(edgeFactor, 2.0) * flicker * angularVariation;
        edgeColor *= 0.7 + 0.5 * ringPattern; // Add ring patterns
        
        // Apply edge glow
        return mix(edgeColor, color, smoothstep(0.0, 1.0, (distToCenter - currentRadius) / 0.015));
    }
    
    return color;
}

// -----------------------------------------------------------------------------
// MAIN - COMPLETELY REWRITTEN FOR GENUINELY SCARY BLACK HOLE
// -----------------------------------------------------------------------------
void main()
{
    // Create nearly pitch black space background
    vec3 background = createVoidBackground(TexCoords);
    
    // Render the accretion disk with cold, alien appearance
    vec3 diskColor = renderAlienDisk(TexCoords);
    
    // Apply extreme gravitational lensing
    vec3 lensedColor = simulateExtremeLensing(TexCoords, diskColor + background);
    
    // Combine disk and background
    vec3 color = lensedColor + diskColor;
    
    // Apply light absorption streaks
    color = applyLightAbsorption(color, TexCoords);
    
    // Apply eerie bloom effect
    color = applyEerieBloom(color, TexCoords, bloomStrength * 2.0);
    
    // Apply exposure adjustment
    color *= exposure * 0.9; // Slightly darker overall
    
    // Apply saturation adjustment - desaturate slightly for colder look
    color = mix(vec3(dot(color, vec3(0.2126, 0.7152, 0.0722))), color, saturation * 0.9);
    
    // Apply extreme contrast tone mapping
    color = adjustContrast(color, contrast * 1.3); // More contrast
    color = extremeContrastToneMapping(color);
    
    // Always apply dark vignette for closing-in feeling
    color = applyExtremeDarkVignette(color, TexCoords);
    
    // Final gamma correction with power curve for darker shadows
    color = pow(max(color, vec3(0.0)), vec3(1.0 / (gamma * 0.85)));
    
    // Create unstable void at event horizon with energy discharge
    color = createUnstableVoid(color, TexCoords);
    
    // Occasional distant "energy arcs" in the distance
    vec2 center = vec2(0.5, 0.5);
    float dist = length(TexCoords - center);
    
    // Energy arcs in outer regions
    if (dist > 0.4 && dist < 0.7) {
        float arcTimingA = sin(time * 0.23) * sin(time * 0.37);
        float arcTimingB = sin(time * 0.17 + 1.5) * sin(time * 0.31 + 0.8);
        
        // First occasional arc set
        if (arcTimingA > 0.9) {
            float arcFactorA = (arcTimingA - 0.9) * 10.0;
            float angle = atan(TexCoords.y - center.y, TexCoords.x - center.x);
            float arcPatternA = smoothstep(0.48, 0.52, sin(angle * 8.0 + dist * 30.0 + time * 2.0));
            
            // Electric blue energy arcs
            color += vec3(0.0, 0.3, 0.8) * arcPatternA * arcFactorA * 0.5 * (1.0 - (dist - 0.4) / 0.3);
        }
        
        // Second set with different pattern
        if (arcTimingB > 0.93) {
            float arcFactorB = (arcTimingB - 0.93) * 14.0;
            float angleB = atan(TexCoords.y - center.y, TexCoords.x - center.x);
            float arcPatternB = smoothstep(0.48, 0.52, sin(angleB * 12.0 - dist * 40.0 - time * 3.0));
            
            // Different blue tone for variety
            color += vec3(0.1, 0.2, 0.7) * arcPatternB * arcFactorB * 0.4 * (1.0 - (dist - 0.4) / 0.3);
        }
    }
    
    // Ensure final color is properly clamped
    color = clamp(color, 0.0, 1.0);
    
    FragColor = vec4(color, 1.0);
}