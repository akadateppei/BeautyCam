import simd

struct FaceBounds {
    let minX: Float
    let maxX: Float
    let minY: Float
    let maxY: Float
    let minZ: Float
    let maxZ: Float

    let centerX: Float
    let centerY: Float
    let centerZ: Float

    let width: Float
    let height: Float
    let depth: Float

    init(vertices: [simd_float3]) {
        var minX = Float.greatestFiniteMagnitude
        var minY = Float.greatestFiniteMagnitude
        var minZ = Float.greatestFiniteMagnitude
        var maxX = -Float.greatestFiniteMagnitude
        var maxY = -Float.greatestFiniteMagnitude
        var maxZ = -Float.greatestFiniteMagnitude

        for v in vertices {
            minX = Swift.min(minX, v.x)
            minY = Swift.min(minY, v.y)
            minZ = Swift.min(minZ, v.z)
            maxX = Swift.max(maxX, v.x)
            maxY = Swift.max(maxY, v.y)
            maxZ = Swift.max(maxZ, v.z)
        }

        self.minX = minX; self.maxX = maxX
        self.minY = minY; self.maxY = maxY
        self.minZ = minZ; self.maxZ = maxZ

        self.centerX = (minX + maxX) * 0.5
        self.centerY = (minY + maxY) * 0.5
        self.centerZ = (minZ + maxZ) * 0.5

        self.width  = max(maxX - minX, 0.0001)
        self.height = max(maxY - minY, 0.0001)
        self.depth  = max(maxZ - minZ, 0.0001)
    }

    func normalized(_ v: simd_float3) -> simd_float2 {
        return simd_float2(
            (v.x - centerX) / width,
            (v.y - centerY) / height
        )
    }
}
