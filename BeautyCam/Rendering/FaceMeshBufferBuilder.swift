import Metal
import simd

/// Flat struct matching the Metal vertex descriptor (no padding issues).
struct FaceVertex {
    var px, py, pz: Float   // position – 12 bytes at offset 0
    var u, v: Float          // uv       –  8 bytes at offset 12
    // stride = 20

    static let stride = 20
}

final class FaceMeshBufferBuilder {
    private var vertexBuffer: MTLBuffer?
    private var indexBuffer: MTLBuffer?
    private var lastVertexCount = 0

    func buildVertexBuffer(
        vertices: [simd_float3],
        uvs: [vector_float2],
        device: MTLDevice
    ) -> MTLBuffer? {
        let count = min(vertices.count, uvs.count)
        guard count > 0 else { return nil }

        // Reuse existing buffer if size matches
        if vertexBuffer == nil || lastVertexCount != count {
            vertexBuffer = device.makeBuffer(
                length: count * FaceVertex.stride,
                options: .storageModeShared
            )
            lastVertexCount = count
        }
        guard let buf = vertexBuffer else { return nil }

        let ptr = buf.contents().bindMemory(to: FaceVertex.self, capacity: count)
        for i in 0..<count {
            ptr[i] = FaceVertex(
                px: vertices[i].x, py: vertices[i].y, pz: vertices[i].z,
                u: uvs[i].x,       v: uvs[i].y
            )
        }
        return buf
    }

    func buildIndexBuffer(indices: [Int16], device: MTLDevice) -> MTLBuffer? {
        if indexBuffer != nil { return indexBuffer }
        let length = indices.count * MemoryLayout<Int16>.stride
        indexBuffer = indices.withUnsafeBytes { ptr in
            device.makeBuffer(bytes: ptr.baseAddress!, length: length, options: .storageModeShared)
        }
        return indexBuffer
    }
}
