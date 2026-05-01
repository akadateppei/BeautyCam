import Metal
import MetalKit
import ARKit
import UIKit
import simd

final class MetalFaceRenderer: NSObject {
    private let device: MTLDevice
    private let commandQueue: MTLCommandQueue
    private let backgroundPipeline: MTLRenderPipelineState
    private let faceMeshPipeline: MTLRenderPipelineState
    private let wireframePipeline: MTLRenderPipelineState
    private let samplerState: MTLSamplerState
    private let textureCache: CameraTextureCache
    private let bufferBuilder = FaceMeshBufferBuilder()
    private let warpController = FaceWarpController()

    let sessionManager: ARFaceSessionManager
    var parameters: BeautyParameters = .default
    var showWireframe: Bool = false

    private var wasFaceTracked = false

    func reset() {
        warpController.reset()
    }

    init(device: MTLDevice, sessionManager: ARFaceSessionManager) throws {
        self.device = device
        self.sessionManager = sessionManager

        guard let queue = device.makeCommandQueue() else {
            throw RendererError.deviceSetupFailed
        }
        self.commandQueue = queue

        guard let cache = CameraTextureCache(device: device) else {
            throw RendererError.deviceSetupFailed
        }
        self.textureCache = cache

        guard let library = device.makeDefaultLibrary() else {
            throw RendererError.shaderLoadFailed
        }

        let pixelFormat: MTLPixelFormat = .bgra8Unorm

        // Background pipeline – vertex shader takes no buffer; fragment shader applies slim warp
        let bgDesc = MTLRenderPipelineDescriptor()
        bgDesc.vertexFunction   = library.makeFunction(name: "backgroundVertexShader")
        bgDesc.fragmentFunction = library.makeFunction(name: "cameraFragmentShader")
        bgDesc.colorAttachments[0].pixelFormat = pixelFormat
        self.backgroundPipeline = try device.makeRenderPipelineState(descriptor: bgDesc)

        // Face mesh vertex descriptor
        let vd = MTLVertexDescriptor()
        vd.attributes[0].format = .float3; vd.attributes[0].offset = 0;  vd.attributes[0].bufferIndex = 0
        vd.attributes[1].format = .float2; vd.attributes[1].offset = 12; vd.attributes[1].bufferIndex = 0
        vd.layouts[0].stride = FaceVertex.stride
        vd.layouts[0].stepFunction = .perVertex

        // Face mesh solid pipeline
        let meshDesc = MTLRenderPipelineDescriptor()
        meshDesc.vertexFunction   = library.makeFunction(name: "faceVertexShader")
        meshDesc.fragmentFunction = library.makeFunction(name: "faceFragmentShader")
        meshDesc.colorAttachments[0].pixelFormat = pixelFormat
        meshDesc.vertexDescriptor = vd
        self.faceMeshPipeline = try device.makeRenderPipelineState(descriptor: meshDesc)

        // Wireframe pipeline
        let wireDesc = MTLRenderPipelineDescriptor()
        wireDesc.vertexFunction   = library.makeFunction(name: "faceVertexShader")
        wireDesc.fragmentFunction = library.makeFunction(name: "wireframeFragmentShader")
        wireDesc.colorAttachments[0].pixelFormat = pixelFormat
        wireDesc.colorAttachments[0].isBlendingEnabled = true
        wireDesc.colorAttachments[0].sourceRGBBlendFactor = .sourceAlpha
        wireDesc.colorAttachments[0].destinationRGBBlendFactor = .oneMinusSourceAlpha
        wireDesc.vertexDescriptor = vd
        self.wireframePipeline = try device.makeRenderPipelineState(descriptor: wireDesc)

        // Sampler
        let sd = MTLSamplerDescriptor()
        sd.minFilter    = .linear
        sd.magFilter    = .linear
        sd.sAddressMode = .clampToEdge
        sd.tAddressMode = .clampToEdge
        self.samplerState = device.makeSamplerState(descriptor: sd)!

        super.init()
    }

    enum RendererError: Error {
        case deviceSetupFailed
        case shaderLoadFailed
    }
}

// MARK: - MTKViewDelegate
extension MetalFaceRenderer: MTKViewDelegate {
    func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {}

    func draw(in view: MTKView) {
        guard let frame = sessionManager.latestFrame,
              let drawable = view.currentDrawable,
              let rpd = view.currentRenderPassDescriptor,
              let commandBuffer = commandQueue.makeCommandBuffer(),
              let encoder = commandBuffer.makeRenderCommandEncoder(descriptor: rpd) else {
            return
        }

        let viewportSize = view.drawableSize
        let (yTex, cbcrTex) = textureCache.textures(from: frame.capturedImage)

        // Compute slim params from face anchor before drawing background,
        // so the background warp matches the face mesh warp in the same frame.
        let slimParams = buildSlimParams(frame: frame, viewportSize: viewportSize,
                                        anchor: sessionManager.latestFaceAnchor)

        // 1. Background (with face slim warp baked in)
        drawBackground(encoder: encoder, frame: frame, viewportSize: viewportSize,
                       yTexture: yTex, cbcrTexture: cbcrTex, slimParams: slimParams)

        // 2. Face mesh (reset smoother when face tracking is lost)
        if let anchor = sessionManager.latestFaceAnchor {
            wasFaceTracked = true
            warpController.parameters = parameters
            drawFaceMesh(encoder: encoder, frame: frame, anchor: anchor,
                         viewportSize: viewportSize, yTexture: yTex, cbcrTexture: cbcrTex,
                         slimParams: slimParams)
        } else if wasFaceTracked {
            wasFaceTracked = false
            warpController.reset()
        }

        encoder.endEncoding()
        commandBuffer.present(drawable)
        commandBuffer.commit()
    }

    // MARK: Background

    private func drawBackground(
        encoder: MTLRenderCommandEncoder,
        frame: ARFrame,
        viewportSize: CGSize,
        yTexture: MTLTexture?,
        cbcrTexture: MTLTexture?,
        slimParams: FaceSlimUniforms
    ) {
        guard let y = yTexture, let cbcr = cbcrTexture else { return }

        // displayTransform inverted: screen UV → camera UV (used by fragment shader)
        let affine = frame.displayTransform(for: .portrait, viewportSize: viewportSize).inverted()
        var transform = simd_float3x3(affine)
        var slim = slimParams

        encoder.setRenderPipelineState(backgroundPipeline)
        encoder.setFragmentTexture(y,    index: 0)
        encoder.setFragmentTexture(cbcr, index: 1)
        encoder.setFragmentSamplerState(samplerState, index: 0)
        encoder.setFragmentBytes(&transform, length: MemoryLayout<simd_float3x3>.size, index: 0)
        encoder.setFragmentBytes(&slim,      length: MemoryLayout<FaceSlimUniforms>.size, index: 1)
        encoder.drawPrimitives(type: .triangleStrip, vertexStart: 0, vertexCount: 4)
    }

    // MARK: Face mesh

    private func drawFaceMesh(
        encoder: MTLRenderCommandEncoder,
        frame: ARFrame,
        anchor: ARFaceAnchor,
        viewportSize: CGSize,
        yTexture: MTLTexture?,
        cbcrTexture: MTLTexture?,
        slimParams: FaceSlimUniforms
    ) {
        guard let y = yTexture, let cbcr = cbcrTexture else { return }

        let geometry = anchor.geometry
        let originalVertices = Array(geometry.vertices)
        let rawIdx = Array(geometry.triangleIndices)

        let proj    = frame.camera.projectionMatrix(for: .portrait, viewportSize: viewportSize, zNear: 0.001, zFar: 10.0)
        let viewMat = frame.camera.viewMatrix(for: .portrait)
        let mvp     = proj * viewMat * anchor.transform
        var uniforms = FaceMeshUniforms(modelViewProjectionMatrix: mvp)

        let displayInv = simd_float3x3(
            frame.displayTransform(for: .portrait, viewportSize: viewportSize).inverted()
        )

        // For each original vertex: project to screen UV, apply slim warp in screen space,
        // then convert to camera UV.  This keeps the warp in the same coordinate system as
        // the background shader so there is no seam at the mesh boundary.
        let cameraUVs: [SIMD2<Float>] = originalVertices.map { v in
            let clip = mvp * simd_float4(v.x, v.y, v.z, 1.0)
            let w = max(clip.w, 1e-4)
            var su = (clip.x / w + 1.0) * 0.5
            var sv = (1.0 - clip.y / w) * 0.5

            let halfW = slimParams.faceHalfWidthScreenU
            let fullW = halfW * 2.0

            // Face slim (identical to Metal shader)
            if slimParams.slimAmount > 0 && fullW > 0 {
                let dx          = su - slimParams.faceCenterScreenU
                let nx          = abs(dx) / fullW
                let regionW     = smoothstep(0.22, 0.50, nx)
                let falloffW    = 1.0 - smoothstep(0.50, 1.00, nx)
                su += sign(dx) * fullW * 0.045 * regionW * falloffW * slimParams.slimAmount
            }

            // Jaw sharpness (identical to Metal shader)
            let jawH = slimParams.jawBottomScreenV - slimParams.jawStartScreenV
            if slimParams.jawAmount > 0 && jawH > 0 && fullW > 0 {
                let dv = sv - slimParams.jawStartScreenV
                if dv > 0 {
                    let ny2   = min(dv / jawH, 1.0)
                    let vertW = smoothstep(0.0, 0.35, ny2)
                    let dx2   = su - slimParams.faceCenterScreenU
                    let nx2   = abs(dx2) / fullW
                    let sideW = smoothstep(0.10, 0.32, nx2) * (1.0 - smoothstep(0.40, 0.60, nx2))
                    su += sign(dx2) * fullW * 0.06 * vertW * sideW * slimParams.jawAmount
                }
            }

            // Eye enlargement UV: pull toward eye center (identical to Metal shader)
            let eyeRadU = slimParams.eyeRadiusU
            let eyeRadV = slimParams.eyeRadiusV
            if slimParams.eyeScaleAmount > 0 && eyeRadU > 0 {
                let eyeCenters: [(Float, Float)] = [
                    (slimParams.leftEyeU,  slimParams.leftEyeV),
                    (slimParams.rightEyeU, slimParams.rightEyeV)
                ]
                for (eyeU, eyeV) in eyeCenters {
                    let dsu = su - eyeU
                    let dsv = sv - eyeV
                    let ed  = sqrt((dsu / eyeRadU) * (dsu / eyeRadU) + (dsv / eyeRadV) * (dsv / eyeRadV))
                    if ed < 1.0 && ed > 0.001 {
                        let ew = (1.0 - smoothstep(0.0, 1.0, ed)) * slimParams.eyeScaleAmount
                        su -= dsu * ew * 0.20
                        sv -= dsv * ew * 0.20
                    }
                }
            }

            let cam = displayInv * SIMD3<Float>(su, sv, 1.0)
            return SIMD2<Float>(cam.x / cam.z, cam.y / cam.z)
        }

        // Vertex positions: warpController handles eyes, nose, and mouth.
        // Face slim and jaw sharpness are UV-only (above).
        let warped = warpController.warp(vertices: originalVertices)

        guard let vBuf = bufferBuilder.buildVertexBuffer(vertices: warped, uvs: cameraUVs, device: device),
              let iBuf = bufferBuilder.buildIndexBuffer(indices: rawIdx, device: device) else {
            return
        }

        let indexCount = rawIdx.count

        var slim = slimParams

        // Solid face mesh
        encoder.setRenderPipelineState(faceMeshPipeline)
        encoder.setVertexBuffer(vBuf, offset: 0, index: 0)
        encoder.setVertexBytes(&uniforms, length: MemoryLayout<FaceMeshUniforms>.size, index: 1)
        encoder.setFragmentTexture(y,    index: 0)
        encoder.setFragmentTexture(cbcr, index: 1)
        encoder.setFragmentSamplerState(samplerState, index: 0)
        encoder.setFragmentBytes(&slim, length: MemoryLayout<FaceSlimUniforms>.size, index: 0)
        encoder.drawIndexedPrimitives(type: .triangle, indexCount: indexCount,
                                      indexType: .uint16, indexBuffer: iBuf, indexBufferOffset: 0)

        // Wireframe overlay
        if showWireframe {
            encoder.setRenderPipelineState(wireframePipeline)
            encoder.setVertexBuffer(vBuf, offset: 0, index: 0)
            encoder.setVertexBytes(&uniforms, length: MemoryLayout<FaceMeshUniforms>.size, index: 1)
            encoder.setTriangleFillMode(.lines)
            encoder.drawIndexedPrimitives(type: .triangle, indexCount: indexCount,
                                          indexType: .uint16, indexBuffer: iBuf, indexBufferOffset: 0)
            encoder.setTriangleFillMode(.fill)
        }
    }

    // MARK: Slim params

    private func buildSlimParams(
        frame: ARFrame,
        viewportSize: CGSize,
        anchor: ARFaceAnchor?
    ) -> FaceSlimUniforms {
        let slimAmount     = parameters.faceSlim * parameters.overallStrength
        let jawAmount      = parameters.jawSharpness * parameters.overallStrength
        let skinSmooth     = parameters.skinSmooth * parameters.overallStrength
        let eyeScaleAmount = parameters.eyeScale * parameters.overallStrength
        guard let anchor = anchor else {
            return FaceSlimUniforms(
                faceCenterScreenU: 0.5, faceHalfWidthScreenU: 0,
                slimAmount: 0, jawAmount: 0,
                jawStartScreenV: 0, jawBottomScreenV: 0,
                skinSmooth: 0, eyeScaleAmount: 0,
                leftEyeU: 0, leftEyeV: 0, rightEyeU: 0, rightEyeV: 0,
                eyeRadiusU: 0, eyeRadiusV: 0, faceTopScreenV: 0, _pad: 0)
        }
        let proj    = frame.camera.projectionMatrix(for: .portrait, viewportSize: viewportSize, zNear: 0.001, zFar: 10.0)
        let viewMat = frame.camera.viewMatrix(for: .portrait)
        let mvp     = proj * viewMat * anchor.transform

        var minSU: Float = 1, maxSU: Float = 0
        var minSV: Float = 1, maxSV: Float = 0
        for v in Array(anchor.geometry.vertices) {
            let clip = mvp * simd_float4(v.x, v.y, v.z, 1.0)
            let w = max(clip.w, 1e-4)
            let su = (clip.x / w + 1.0) * 0.5
            let sv = (1.0 - clip.y / w) * 0.5
            if su < minSU { minSU = su }
            if su > maxSU { maxSU = su }
            if sv < minSV { minSV = sv }
            if sv > maxSV { maxSV = sv }
        }

        let faceW      = maxSU - minSU
        let faceH      = maxSV - minSV
        let faceCenterU = (minSU + maxSU) * 0.5
        let jawStartV   = minSV + faceH * 0.48
        // Eye centers: ~30% horizontal from center, ~38% down from forehead
        let eyeOffU     = faceW * 0.30
        let eyeV        = minSV + faceH * 0.38
        let eyeRadU     = faceW * 0.14
        let eyeRadV     = faceH * 0.09

        return FaceSlimUniforms(
            faceCenterScreenU:    faceCenterU,
            faceHalfWidthScreenU: faceW * 0.5,
            slimAmount:     slimAmount,
            jawAmount:      jawAmount,
            jawStartScreenV:  jawStartV,
            jawBottomScreenV: maxSV,
            skinSmooth:     skinSmooth,
            eyeScaleAmount: eyeScaleAmount,
            leftEyeU:  faceCenterU - eyeOffU, leftEyeV:  eyeV,
            rightEyeU: faceCenterU + eyeOffU, rightEyeV: eyeV,
            eyeRadiusU: eyeRadU, eyeRadiusV: eyeRadV,
            faceTopScreenV: minSV,
            _pad: 0
        )
    }
}

// MARK: - FaceSlimUniforms (Swift mirror of Metal struct; must match layout exactly)
private struct FaceSlimUniforms {
    var faceCenterScreenU: Float
    var faceHalfWidthScreenU: Float
    var slimAmount: Float
    var jawAmount: Float
    var jawStartScreenV: Float
    var jawBottomScreenV: Float
    var skinSmooth: Float
    var eyeScaleAmount: Float
    var leftEyeU: Float
    var leftEyeV: Float
    var rightEyeU: Float
    var rightEyeV: Float
    var eyeRadiusU: Float
    var eyeRadiusV: Float
    var faceTopScreenV: Float
    var _pad: Float
}

// MARK: - FaceMeshUniforms (Swift mirror of Metal struct)
private struct FaceMeshUniforms {
    var modelViewProjectionMatrix: simd_float4x4
}

// MARK: - CGAffineTransform → simd_float3x3
extension simd_float3x3 {
    init(_ t: CGAffineTransform) {
        self.init(columns: (
            SIMD3<Float>(Float(t.a),  Float(t.b),  0),
            SIMD3<Float>(Float(t.c),  Float(t.d),  0),
            SIMD3<Float>(Float(t.tx), Float(t.ty), 1)
        ))
    }
}
