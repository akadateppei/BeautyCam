#include <metal_stdlib>
using namespace metal;

// ---- Shared structures ----

struct FaceVertexIn {
    float3 position [[attribute(0)]];
    float2 uv       [[attribute(1)]];
};

struct VertexOut {
    float4 position [[position]];
    float2 uv;
    float2 screenUV;  // normalized 0..1 in screen space; used by face pass for eye-shine localization
};

struct FaceMeshUniforms {
    float4x4 modelViewProjectionMatrix;
};

struct FaceSlimUniforms {
    // row 0
    float faceCenterScreenU;
    float faceHalfWidthScreenU;
    float slimAmount;
    float jawAmount;
    // row 1
    float jawStartScreenV;
    float jawBottomScreenV;
    float skinSmooth;
    float eyeScaleAmount;
    // row 2
    float leftEyeU;
    float leftEyeV;
    float rightEyeU;
    float rightEyeV;
    // row 3
    float eyeRadiusU;
    float eyeRadiusV;
    float faceTopScreenV;
    float templeScreenV;
    // row 4
    float eyeShineAmount;
    float _pad0;
    float _pad1;
    float _pad2;
};

// BT.601 full-range YCbCr → RGB, compile-time constant
constant half3x3 kYCbCrToRGB = half3x3(
    half3( 1.0h,    1.0h,      1.0h),
    half3( 0.0h,   -0.344136h, 1.772h),
    half3( 1.402h, -0.714136h, 0.0h)
);

// ---- Helpers ----

// Skin tone in full-range CbCr — wider range to include more skin pixels at full strength
half skinWeight(half2 cbcr) {
    half cbMask = smoothstep(0.37h, 0.42h, cbcr.r) * (1.0h - smoothstep(0.51h, 0.56h, cbcr.r));
    half crMask = smoothstep(0.49h, 0.53h, cbcr.g) * (1.0h - smoothstep(0.61h, 0.66h, cbcr.g));
    return cbMask * crMask;
}

// 13-tap cross blur on Y with three radii — much wider, smoother skin softening.
// Rings at 2, 5, 10 texels. Center weight kept low so the result is dominated by neighbors.
half blurY(texture2d<half> tex, sampler s, float2 uv, float2 texelSize) {
    float2 p1 = texelSize * 2.5;
    float2 p2 = texelSize * 6.0;
    float2 p3 = texelSize * 11.0;
    half c   = tex.sample(s, uv).r;
    half i1  = tex.sample(s, uv + float2( p1.x,    0)).r
             + tex.sample(s, uv + float2(-p1.x,    0)).r
             + tex.sample(s, uv + float2(    0, p1.y)).r
             + tex.sample(s, uv + float2(    0,-p1.y)).r;
    half i2  = tex.sample(s, uv + float2( p2.x,    0)).r
             + tex.sample(s, uv + float2(-p2.x,    0)).r
             + tex.sample(s, uv + float2(    0, p2.y)).r
             + tex.sample(s, uv + float2(    0,-p2.y)).r;
    half i3  = tex.sample(s, uv + float2( p3.x,    0)).r
             + tex.sample(s, uv + float2(-p3.x,    0)).r
             + tex.sample(s, uv + float2(    0, p3.y)).r
             + tex.sample(s, uv + float2(    0,-p3.y)).r;
    // 0.16 + 4·0.12 + 4·0.06 + 4·0.03 = 0.16 + 0.48 + 0.24 + 0.12 = 1.00
    return c * 0.16h + i1 * 0.12h + i2 * 0.06h + i3 * 0.03h;
}

// Edge-aware skin blend: reduce blend weight only on STRONG edges (eyes/lips/eyebrows).
// Looser threshold so subtle skin texture (pores, fine lines) still gets smoothed.
half edgeWeight(half centerY, half blurredY) {
    half d = abs(centerY - blurredY);
    return 1.0h - smoothstep(0.12h, 0.32h, d);
}

half4 toRGBA(half y, half2 cbcr) {
    half3 rgb = clamp(kYCbCrToRGB * half3(y, cbcr.x - 0.5h, cbcr.y - 0.5h), 0.0h, 1.0h);
    return half4(rgb, 1.0h);
}

// ---- Background pass ----

vertex VertexOut backgroundVertexShader(uint vid [[vertex_id]]) {
    constexpr float2 positions[4] = {
        float2(-1.0,  1.0), float2(-1.0, -1.0),
        float2( 1.0,  1.0), float2( 1.0, -1.0),
    };
    constexpr float2 screenUVs[4] = {
        float2(0.0, 0.0), float2(0.0, 1.0),
        float2(1.0, 0.0), float2(1.0, 1.0),
    };
    VertexOut out;
    out.position = float4(positions[vid], 0.0, 1.0);
    out.uv = screenUVs[vid];
    out.screenUV = screenUVs[vid];
    return out;
}

fragment half4 cameraFragmentShader(
    VertexOut in [[stage_in]],
    texture2d<half> yTexture    [[texture(0)]],
    texture2d<half> cbcrTexture [[texture(1)]],
    sampler s [[sampler(0)]],
    constant float3x3& displayTransform [[buffer(0)]],
    constant FaceSlimUniforms& slim [[buffer(1)]]
) {
    float2 screenUV = in.uv;
    float  fullW    = slim.faceHalfWidthScreenU * 2.0;

    // Face slim — keep head size unchanged; ramp from temples down to chin
    if (slim.slimAmount > 0.0 && fullW > 0.0 && slim.jawBottomScreenV > slim.templeScreenV) {
        float dx     = screenUV.x - slim.faceCenterScreenU;
        float nx     = abs(dx) / fullW;
        float vGrad  = smoothstep(slim.templeScreenV, slim.jawBottomScreenV, screenUV.y);
        float weight = smoothstep(0.22, 0.50, nx)
                     * (1.0 - smoothstep(0.50, 1.00, nx))
                     * vGrad
                     * slim.slimAmount;
        screenUV.x  += sign(dx) * fullW * 0.045 * weight;
    }

    // Jaw sharpness: lower face sides
    if (slim.jawAmount > 0.0 && slim.jawBottomScreenV > slim.jawStartScreenV) {
        float jawH = slim.jawBottomScreenV - slim.jawStartScreenV;
        float dv   = screenUV.y - slim.jawStartScreenV;
        if (dv > 0.0 && fullW > 0.0) {
            float ny2    = clamp(dv / jawH, 0.0, 1.0);
            float dx2    = screenUV.x - slim.faceCenterScreenU;
            float nx2    = abs(dx2) / fullW;
            float weight = smoothstep(0.0, 0.35, ny2)
                         * smoothstep(0.10, 0.32, nx2)
                         * (1.0 - smoothstep(0.40, 0.60, nx2))
                         * slim.jawAmount;
            screenUV.x  += sign(dx2) * fullW * 0.06 * weight;
        }
    }

    // Eye scale: local UV compression toward each eye center → eye appears larger
    // Quadratic (1-n)^2 falloff concentrates the effect at the iris, fading cleanly at radius
    if (slim.eyeScaleAmount > 0.0 && slim.eyeRadiusU > 0.0) {
        float  pull   = slim.eyeScaleAmount * 0.13;
        float2 eyeRad = float2(slim.eyeRadiusU, slim.eyeRadiusV) * 1.4;
        float2 orig   = screenUV;

        float2 dL = orig - float2(slim.leftEyeU,  slim.leftEyeV);
        float  nL = clamp(length(dL / eyeRad), 0.0, 1.0);
        float  wL = pull * (1.0 - nL) * (1.0 - nL);

        float2 dR = orig - float2(slim.rightEyeU, slim.rightEyeV);
        float  nR = clamp(length(dR / eyeRad), 0.0, 1.0);
        float  wR = pull * (1.0 - nR) * (1.0 - nR);

        screenUV -= dL * wL + dR * wR;
    }

    // Screen UV → camera UV (displayTransform is affine; z is always 1)
    float2 camUV = (displayTransform * float3(screenUV, 1.0)).xy;

    // Background pass: no skin smoothing here — that's the face-mesh pass's job,
    // so the effect stays strictly within the face area.
    half  y    = yTexture.sample(s, camUV).r;
    half2 cbcr = cbcrTexture.sample(s, camUV).rg;
    return toRGBA(y, cbcr);
}

// ---- Face mesh pass ----

vertex VertexOut faceVertexShader(
    FaceVertexIn in [[stage_in]],
    constant FaceMeshUniforms& uniforms [[buffer(1)]]
) {
    VertexOut out;
    out.position = uniforms.modelViewProjectionMatrix * float4(in.position, 1.0);
    out.uv = in.uv;
    // NDC → screen UV (origin top-left): used to locate eyes for shine boost
    float w = max(out.position.w, 1e-4);
    out.screenUV = float2((out.position.x / w + 1.0) * 0.5,
                          (1.0 - out.position.y / w) * 0.5);
    return out;
}

fragment half4 faceFragmentShader(
    VertexOut in [[stage_in]],
    texture2d<half> yTexture    [[texture(0)]],
    texture2d<half> cbcrTexture [[texture(1)]],
    sampler s [[sampler(0)]],
    constant FaceSlimUniforms& slim [[buffer(0)]]
) {
    half  y    = yTexture.sample(s, in.uv).r;
    half2 cbcr = cbcrTexture.sample(s, in.uv).rg;

    // Edge-aware skin smoothing: CbCr mask × edge mask × user strength
    if (slim.skinSmooth > 0.0) {
        float2 texelSize = 1.0 / float2(yTexture.get_width(), yTexture.get_height());
        half   blurred   = blurY(yTexture, s, in.uv, texelSize);
        half   blendW    = skinWeight(cbcr)
                         * edgeWeight(y, blurred)
                         * half(slim.skinSmooth);
        y = mix(y, blurred, blendW);
    }

    // Eye shine: push bright pixels toward white, but only within the eye region.
    // Quadratic radial mask × highlight mask above threshold.
    if (slim.eyeShineAmount > 0.0 && slim.eyeRadiusU > 0.0) {
        float2 rad = float2(slim.eyeRadiusU, slim.eyeRadiusV);
        float2 dL  = in.screenUV - float2(slim.leftEyeU,  slim.leftEyeV);
        float2 dR  = in.screenUV - float2(slim.rightEyeU, slim.rightEyeV);
        float  nL  = clamp(length(dL / rad), 0.0, 1.0);
        float  nR  = clamp(length(dR / rad), 0.0, 1.0);
        // (1-n)^2 — strongest at center, zero at radius
        float  eyeRegion = max((1.0 - nL) * (1.0 - nL), (1.0 - nR) * (1.0 - nR));

        // Highlight mask: ramps in starting at threshold 0.72, saturated past 0.95
        half hl = smoothstep(0.72h, 0.95h, y);
        half boost = hl * half(eyeRegion) * half(slim.eyeShineAmount);
        // Push toward 1.0 without clipping (asymptotic)
        y = y + (1.0h - y) * boost;
    }

    return toRGBA(y, cbcr);
}

// ---- Wireframe pass ----

fragment half4 wireframeFragmentShader(VertexOut in [[stage_in]]) {
    return half4(0.0h, 1.0h, 0.5h, 0.85h);
}
