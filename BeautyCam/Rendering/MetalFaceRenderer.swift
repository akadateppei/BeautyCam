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

        // Background pipeline (no vertex descriptor; vertex_id based)
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

        // 1. Background
        drawBackground(encoder: encoder, frame: frame, viewportSize: viewportSize,
                       yTexture: yTex, cbcrTexture: cbcrTex)

        // 2. Face mesh (reset smoother when face tracking is lost)
        if let anchor = sessionManager.latestFaceAnchor {
            wasFaceTracked = true
            warpController.parameters = parameters
            drawFaceMesh(encoder: encoder, frame: frame, anchor: anchor,
                         viewportSize: viewportSize, yTexture: yTex, cbcrTexture: cbcrTex)
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
        cbcrTexture: MTLTexture?
    ) {
        guard let y = yTexture, let cbcr = cbcrTexture else { return }

        // displayTransform maps camera→display; invert to get display UV → camera UV for sampling
        let affine = frame.displayTransform(for: .portrait, viewportSize: viewportSize).inverted()
        var transform = simd_float3x3(affine)

        encoder.setRenderPipelineState(backgroundPipeline)
        encoder.setFragmentTexture(y,    index: 0)
        encoder.setFragmentTexture(cbcr, index: 1)
        encoder.setFragmentSamplerState(samplerState, index: 0)
        encoder.setVertexBytes(&transform, length: MemoryLayout<simd_float3x3>.size, index: 1)
        encoder.drawPrimitives(type: .triangleStrip, vertexStart: 0, vertexCount: 4)
    }

    // MARK: Face mesh

    private func drawFaceMesh(
        encoder: MTLRenderCommandEncoder,
        frame: ARFrame,
        anchor: ARFaceAnchor,
        viewportSize: CGSize,
        yTexture: MTLTexture?,
        cbcrTexture: MTLTexture?
    ) {
        guard let y = yTexture, let cbcr = cbcrTexture else { return }

        let geometry = anchor.geometry
        let originalVertices = Array(geometry.vertices)
        let uvs    = Array(geometry.textureCoordinates)
        // triangleIndices is UnsafePointer<Int16>; each triangle = 3 indices
        let rawIdx = Array(UnsafeBufferPointer(start: geometry.triangleIndices,
                                               count: geometry.triangleCount * 3))

        let warped = warpController.warp(vertices: originalVertices)

        guard let vBuf = bufferBuilder.buildVertexBuffer(vertices: warped, uvs: uvs, device: device),
              let iBuf = bufferBuilder.buildIndexBuffer(indices: rawIdx, device: device) else {
            return
        }

        let proj = frame.camera.projectionMatrix(for: .portrait, viewportSize: viewportSize, zNear: 0.001, zFar: 10.0)
        let view = frame.camera.viewMatrix(for: .portrait)
        var uniforms = FaceMeshUniforms(modelViewProjectionMatrix: proj * view * anchor.transform)

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
