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

// Face slim is applied in screen UV space so the background and face mesh warp identically.
struct FaceSlimUniforms {
    float faceCenterScreenU;    // screen U of face center (0–1)
    float faceHalfWidthScreenU; // face half-width in screen U
    float slimAmount;           // 0 = off, 1 = max
    float _pad;
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
// Vertex shader outputs screen UV; the fragment shader applies the slim warp
// in screen space, then converts to camera UV via displayTransform.

vertex VertexOut backgroundVertexShader(uint vid [[vertex_id]]) {
    const float2 positions[4] = {
        float2(-1.0,  1.0),
        float2(-1.0, -1.0),
        float2( 1.0,  1.0),
        float2( 1.0, -1.0),
    };
    // screen UV: (0,0) top-left, (1,1) bottom-right
    const float2 screenUVs[4] = {
        float2(0.0, 0.0),
        float2(0.0, 1.0),
        float2(1.0, 0.0),
        float2(1.0, 1.0),
    };
    VertexOut out;
    out.position = float4(positions[vid], 0.0, 1.0);
    out.uv = screenUVs[vid];
    return out;
}

fragment float4 cameraFragmentShader(
    VertexOut in [[stage_in]],
    texture2d<float> yTexture    [[texture(0)]],
    texture2d<float> cbcrTexture [[texture(1)]],
    sampler s [[sampler(0)]],
    constant float3x3& displayTransform [[buffer(0)]],  // inverted: screen UV → camera UV
    constant FaceSlimUniforms& slim [[buffer(1)]]
) {
    float2 screenUV = in.uv;

    // Face slim warp in screen space – same formula as Swift side so boundary is seamless
    if (slim.slimAmount > 0.0 && slim.faceHalfWidthScreenU > 0.0) {
        float dx = screenUV.x - slim.faceCenterScreenU;
        float nx = dx / (slim.faceHalfWidthScreenU * 2.0);  // –0.5…+0.5 across face
        float sideDistance = abs(nx);
        float regionWeight  = smoothstep(0.22, 0.50, sideDistance);
        float falloffWeight = 1.0 - smoothstep(0.50, 1.00, sideDistance); // fade outside face
        float weight = regionWeight * falloffWeight * slim.slimAmount;
        screenUV.x += sign(dx) * slim.faceHalfWidthScreenU * 2.0 * 0.045 * weight;
    }

    // Convert screen UV → camera UV
    float3 cam3 = displayTransform * float3(screenUV, 1.0);
    float2 camUV = cam3.xy;

    float  y    = yTexture.sample(s, camUV).r;
    float2 cbcr = cbcrTexture.sample(s, camUV).rg;
    return float4(ycbcrToRGB(y, cbcr), 1.0);
}

// ---- Face mesh pass ----

vertex VertexOut faceVertexShader(
    FaceVertexIn in [[stage_in]],
    constant FaceMeshUniforms& uniforms [[buffer(1)]]
) {
    VertexOut out;
    out.position = uniforms.modelViewProjectionMatrix * float4(in.position, 1.0);
    out.uv = in.uv;  // camera UV pre-warped on the CPU side
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

fragment float4 wireframeFragmentShader(VertexOut in [[stage_in]]) {
    return float4(0.0, 1.0, 0.5, 0.85);
}
