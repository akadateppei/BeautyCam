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

// Face slim and jaw are applied in screen UV space so background and face mesh warp identically.
struct FaceSlimUniforms {
    // --- row 0 ---
    float faceCenterScreenU;    // screen U of face center (0–1)
    float faceHalfWidthScreenU; // face half-width in screen U
    float slimAmount;           // face slim (0=off, 1=max)
    float jawAmount;            // jaw (chin tip) sharpness
    // --- row 1 ---
    float jawStartScreenV;      // screen V where jaw region begins (~face midpoint)
    float jawBottomScreenV;     // screen V of chin bottom
    float skinSmooth;           // skin smoothing strength
    float eyeScaleAmount;       // eye enlargement (0=off, 1=max)
    // --- row 2 ---
    float leftEyeU;             // left eye center screen U
    float leftEyeV;             // left eye center screen V
    float rightEyeU;
    float rightEyeV;
    // --- row 3 ---
    float eyeRadiusU;           // eye effect ellipse horizontal radius (screen U)
    float eyeRadiusV;           // eye effect ellipse vertical radius (screen V)
    float faceTopScreenV;       // screen V of forehead top (slim starts here)
    float _pad;
};

// ---- Skin detection + smoothing ----
// Skin tone in CbCr space (full-range): Cb 0.37–0.54, Cr 0.50–0.65
float skinWeight(float2 cbcr) {
    float cb = cbcr.r, cr = cbcr.g;
    float cbMask = smoothstep(0.37, 0.42, cb) * (1.0 - smoothstep(0.51, 0.56, cb));
    float crMask = smoothstep(0.49, 0.53, cr) * (1.0 - smoothstep(0.62, 0.67, cr));
    return cbMask * crMask;
}

// 5-tap cross blur on Y channel; blurRadius in texel units
float blurY(texture2d<float> tex, sampler s, float2 uv, float blurRadius) {
    float2 px = blurRadius / float2(tex.get_width(), tex.get_height());
    return tex.sample(s, uv).r                       * 0.40
         + tex.sample(s, uv + float2( px.x,    0)).r * 0.15
         + tex.sample(s, uv + float2(-px.x,    0)).r * 0.15
         + tex.sample(s, uv + float2(   0,  px.y)).r * 0.15
         + tex.sample(s, uv + float2(   0, -px.y)).r * 0.15;
}

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

    float fullW = slim.faceHalfWidthScreenU * 2.0;

    // Face slim
    if (slim.slimAmount > 0.0 && fullW > 0.0) {
        float dx = screenUV.x - slim.faceCenterScreenU;
        float nx = abs(dx) / fullW;
        float regionWeight  = smoothstep(0.22, 0.50, nx);
        float falloffWeight = 1.0 - smoothstep(0.50, 1.00, nx);
        float weight = regionWeight * falloffWeight * slim.slimAmount;
        screenUV.x += sign(dx) * fullW * 0.045 * weight;
    }

    // Jaw sharpness: lower face sides
    if (slim.jawAmount > 0.0 && slim.jawBottomScreenV > slim.jawStartScreenV) {
        float jawH = slim.jawBottomScreenV - slim.jawStartScreenV;
        float dv   = screenUV.y - slim.jawStartScreenV;
        if (dv > 0.0 && fullW > 0.0) {
            float ny2     = clamp(dv / jawH, 0.0, 1.0);
            float vertW   = smoothstep(0.0, 0.35, ny2);
            float dx2     = screenUV.x - slim.faceCenterScreenU;
            float nx2     = abs(dx2) / fullW;
            float sideW   = smoothstep(0.10, 0.32, nx2) * (1.0 - smoothstep(0.40, 0.60, nx2));
            float weight2 = vertW * sideW * slim.jawAmount;
            screenUV.x += sign(dx2) * fullW * 0.06 * weight2;
        }
    }

    // Eye enlargement: shift UV toward each eye center (UV-based, no mesh gap)
    if (slim.eyeScaleAmount > 0.0 && slim.eyeRadiusU > 0.0) {
        float2 eyeCenters[2] = {float2(slim.leftEyeU,  slim.leftEyeV),
                                float2(slim.rightEyeU, slim.rightEyeV)};
        for (int i = 0; i < 2; i++) {
            float2 dUV  = screenUV - eyeCenters[i];
            float  normX = dUV.x / slim.eyeRadiusU;
            float  normY = dUV.y / slim.eyeRadiusV;
            float  ed    = sqrt(normX * normX + normY * normY);
            if (ed < 1.0 && ed > 0.001) {
                float ew = (1.0 - smoothstep(0.0, 1.0, ed)) * slim.eyeScaleAmount;
                screenUV -= dUV * ew * 0.20;  // pull UV toward eye center → iris appears larger
            }
        }
    }

    // Convert screen UV → camera UV
    float3 cam3 = displayTransform * float3(screenUV, 1.0);
    float2 camUV = cam3.xy;

    float  y    = yTexture.sample(s, camUV).r;
    float2 cbcr = cbcrTexture.sample(s, camUV).rg;

    // Skin smoothing: blur Y (luminance) in skin-tone regions, preserve CbCr for color
    if (slim.skinSmooth > 0.0) {
        float sw = skinWeight(cbcr);
        if (sw > 0.0) {
            float yBlurred = blurY(yTexture, s, camUV, 3.5);
            y = mix(y, yBlurred, sw * slim.skinSmooth);
        }
    }

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
    sampler s [[sampler(0)]],
    constant FaceSlimUniforms& slim [[buffer(0)]]
) {
    float  y    = yTexture.sample(s, in.uv).r;
    float2 cbcr = cbcrTexture.sample(s, in.uv).rg;

    if (slim.skinSmooth > 0.0) {
        float sw = skinWeight(cbcr);
        if (sw > 0.0) {
            float yBlurred = blurY(yTexture, s, in.uv, 3.5);
            y = mix(y, yBlurred, sw * slim.skinSmooth);
        }
    }

    return float4(ycbcrToRGB(y, cbcr), 1.0);
}

// ---- Wireframe pass ----

fragment float4 wireframeFragmentShader(VertexOut in [[stage_in]]) {
    return float4(0.0, 1.0, 0.5, 0.85);
}
