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
    float _pad;
};

// BT.601 full-range YCbCr → RGB, compile-time constant
constant half3x3 kYCbCrToRGB = half3x3(
    half3( 1.0h,    1.0h,      1.0h),
    half3( 0.0h,   -0.344136h, 1.772h),
    half3( 1.402h, -0.714136h, 0.0h)
);

// ---- Helpers ----

// Skin tone in full-range CbCr: Cb 0.37–0.56, Cr 0.49–0.67
half skinWeight(half2 cbcr) {
    half cbMask = smoothstep(0.37h, 0.42h, cbcr.r) * (1.0h - smoothstep(0.51h, 0.56h, cbcr.r));
    half crMask = smoothstep(0.49h, 0.53h, cbcr.g) * (1.0h - smoothstep(0.62h, 0.67h, cbcr.g));
    return cbMask * crMask;
}

// 5-tap cross blur on Y; texelSize = 1/textureResolution, radius in texels
half blurY(texture2d<half> tex, sampler s, float2 uv, float2 texelSize) {
    float2 px = texelSize * 3.5;
    return tex.sample(s, uv).r                      * 0.40h
         + tex.sample(s, uv + float2( px.x,   0)).r * 0.15h
         + tex.sample(s, uv + float2(-px.x,   0)).r * 0.15h
         + tex.sample(s, uv + float2(  0,  px.y)).r * 0.15h
         + tex.sample(s, uv + float2(  0, -px.y)).r * 0.15h;
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

    // Face slim
    if (slim.slimAmount > 0.0 && fullW > 0.0) {
        float dx     = screenUV.x - slim.faceCenterScreenU;
        float nx     = abs(dx) / fullW;
        float weight = smoothstep(0.22, 0.50, nx)
                     * (1.0 - smoothstep(0.50, 1.00, nx))
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

    // Screen UV → camera UV (displayTransform is affine; z is always 1)
    float2 camUV = (displayTransform * float3(screenUV, 1.0)).xy;

    half  y    = yTexture.sample(s, camUV).r;
    half2 cbcr = cbcrTexture.sample(s, camUV).rg;

    // Skin smoothing: blend blurred Y into skin regions; weight=0 outside skin or when disabled
    if (slim.skinSmooth > 0.0) {
        float2 texelSize = 1.0 / float2(yTexture.get_width(), yTexture.get_height());
        half   blendW    = skinWeight(cbcr) * half(slim.skinSmooth);
        y = mix(y, blurY(yTexture, s, camUV, texelSize), blendW);
    }

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

    if (slim.skinSmooth > 0.0) {
        float2 texelSize = 1.0 / float2(yTexture.get_width(), yTexture.get_height());
        half   blendW    = skinWeight(cbcr) * half(slim.skinSmooth);
        y = mix(y, blurY(yTexture, s, in.uv, texelSize), blendW);
    }

    return toRGBA(y, cbcr);
}

// ---- Wireframe pass ----

fragment half4 wireframeFragmentShader(VertexOut in [[stage_in]]) {
    return half4(0.0h, 1.0h, 0.5h, 0.85h);
}
