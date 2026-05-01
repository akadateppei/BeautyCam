import simd

final class TemporalSmoother {
    private var previousVertices: [simd_float3] = []
    var smoothingFactor: Float = 0.55

    func smooth(_ vertices: [simd_float3]) -> [simd_float3] {
        guard previousVertices.count == vertices.count else {
            previousVertices = vertices
            return vertices
        }
        var result = vertices
        for i in vertices.indices {
            result[i] = previousVertices[i] * smoothingFactor + vertices[i] * (1.0 - smoothingFactor)
        }
        previousVertices = result
        return result
    }

    func reset() {
        previousVertices.removeAll()
    }
}
