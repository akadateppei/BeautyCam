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

// ---- Y/CbCr → RGB helper ----
// BT.601 full-range (kCVPixelFormatType_420YpCbCr8BiPlanarFullRange)
float3 ycbcrToRGB(float y, float2 cbcr) {
    float3 ycbcr = float3(y, cbcr.x - 0.5, cbcr.y - 0.5);
    float3x3 m = float3x3(
        float3(1.0,  1.0,    1.0),
        float3(0.0, -0.344136,  1.772),
        float3(1.402, -0.714136, 0.0)
    );
    return clamp(m * ycbcr, 0.0, 1.0);
}

// ---- Background pass ----
// Renders a full-screen quad and applies displayTransform to camera UV.

vertex VertexOut backgroundVertexShader(
    uint vid [[vertex_id]],
    constant float3x3& displayTransform [[buffer(1)]]
) {
    // Triangle strip: TL, BL, TR, BR
    const float2 positions[4] = {
        float2(-1.0,  1.0),
        float2(-1.0, -1.0),
        float2( 1.0,  1.0),
        float2( 1.0, -1.0),
    };
    // Screen UV (0,0) at top-left, (1,1) at bottom-right
    const float2 screenUVs[4] = {
        float2(0.0, 0.0),
        float2(0.0, 1.0),
        float2(1.0, 0.0),
        float2(1.0, 1.0),
    };

    VertexOut out;
    out.position = float4(positions[vid], 0.0, 1.0);
    float3 mapped = displayTransform * float3(screenUVs[vid], 1.0);
    out.uv = mapped.xy;
    return out;
}

fragment float4 cameraFragmentShader(
    VertexOut in [[stage_in]],
    texture2d<float> yTexture    [[texture(0)]],
    texture2d<float> cbcrTexture [[texture(1)]],
    sampler s [[sampler(0)]]
) {
    float  y    = yTexture.sample(s, in.uv).r;
    float2 cbcr = cbcrTexture.sample(s, in.uv).rg;
    return float4(ycbcrToRGB(y, cbcr), 1.0);
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

fragment float4 faceFragmentShader(
    VertexOut in [[stage_in]],
    texture2d<float> yTexture    [[texture(0)]],
    texture2d<float> cbcrTexture [[texture(1)]],
    sampler s [[sampler(0)]]
) {
    float  y    = yTexture.sample(s, in.uv).r;
    float2 cbcr = cbcrTexture.sample(s, in.uv).rg;
    return float4(ycbcrToRGB(y, cbcr), 1.0);
}

// ---- Wireframe pass ----
// Re-uses faceVertexShader; fragment outputs a fixed colour.

fragment float4 wireframeFragmentShader(VertexOut in [[stage_in]]) {
    return float4(0.0, 1.0, 0.5, 0.85);
}
