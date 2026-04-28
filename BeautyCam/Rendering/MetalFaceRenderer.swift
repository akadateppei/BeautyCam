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
            let sv = (1.0 - clip.y / w) * 0.5

            // Slim warp – identical formula to the Metal fragment shader
            let halfW = slimParams.faceHalfWidthScreenU
            if slimParams.slimAmount > 0 && halfW > 0 {
                let dx = su - slimParams.faceCenterScreenU
                let nx = dx / (halfW * 2.0)
                let sideDistance = abs(nx)
                let regionWeight  = smoothstep(0.22, 0.50, sideDistance)
                let falloffWeight = 1.0 - smoothstep(0.50, 1.00, sideDistance)
                let weight = regionWeight * falloffWeight * slimParams.slimAmount
                su += sign(dx) * halfW * 2.0 * 0.045 * weight
            }

            let cam = displayInv * SIMD3<Float>(su, sv, 1.0)
            return SIMD2<Float>(cam.x / cam.z, cam.y / cam.z)
        }

        // Vertex positions: warpController handles jaw, eyes, nose, mouth, and boundary
        // expansion.  Face slim is handled above as a UV-only operation.
        let warped = warpController.warp(vertices: originalVertices)

        guard let vBuf = bufferBuilder.buildVertexBuffer(vertices: warped, uvs: cameraUVs, device: device),
              let iBuf = bufferBuilder.buildIndexBuffer(indices: rawIdx, device: device) else {
            return
        }

        let indexCount = rawIdx.count

        // Solid face mesh
        encoder.setRenderPipelineState(faceMeshPipeline)
        encoder.setVertexBuffer(vBuf, offset: 0, index: 0)
        encoder.setVertexBytes(&uniforms, length: MemoryLayout<FaceMeshUniforms>.size, index: 1)
        encoder.setFragmentTexture(y,    index: 0)
        encoder.setFragmentTexture(cbcr, index: 1)
        encoder.setFragmentSamplerState(samplerState, index: 0)
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
        let slimAmount = parameters.faceSlim * parameters.overallStrength
        guard let anchor = anchor, slimAmount > 0 else {
            return FaceSlimUniforms(faceCenterScreenU: 0.5, faceHalfWidthScreenU: 0,
                                    slimAmount: 0, _pad: 0)
        }
        let proj    = frame.camera.projectionMatrix(for: .portrait, viewportSize: viewportSize, zNear: 0.001, zFar: 10.0)
        let viewMat = frame.camera.viewMatrix(for: .portrait)
        let mvp     = proj * viewMat * anchor.transform

        var minSU: Float = 1, maxSU: Float = 0
        for v in Array(anchor.geometry.vertices) {
            let clip = mvp * simd_float4(v.x, v.y, v.z, 1.0)
            let w = max(clip.w, 1e-4)
            let su = (clip.x / w + 1.0) * 0.5
            if su < minSU { minSU = su }
            if su > maxSU { maxSU = su }
        }
        return FaceSlimUniforms(
            faceCenterScreenU:    (minSU + maxSU) * 0.5,
            faceHalfWidthScreenU: (maxSU - minSU) * 0.5,
            slimAmount: slimAmount,
            _pad: 0
        )
    }
}

// MARK: - FaceSlimUniforms (Swift mirror of Metal struct; must match layout exactly)
private struct FaceSlimUniforms {
    var faceCenterScreenU: Float
    var faceHalfWidthScreenU: Float
    var slimAmount: Float
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
