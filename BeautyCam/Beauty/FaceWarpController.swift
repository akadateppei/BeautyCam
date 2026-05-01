import simd

final class FaceWarpController {
    var parameters: BeautyParameters = .default
    private let smoother = TemporalSmoother()

    func warp(vertices: [simd_float3]) -> [simd_float3] {
        guard !vertices.isEmpty else { return vertices }
        let bounds = FaceBounds(vertices: vertices)
        var result = vertices
        // Eye, jaw, face slim are UV-based in MetalFaceRenderer; no vertex movement here.
        result = applyChinPoint(result, bounds: bounds)
        result = applyNoseSlim(result, bounds: bounds)
        result = applyNoseWing(result, bounds: bounds)
        result = applyMouthAdjust(result, bounds: bounds)
        result = smoother.smooth(result)
        return result
    }

    func reset() {
        smoother.reset()
    }

    // MARK: - Chin Point
    private func applyChinPoint(_ vertices: [simd_float3], bounds: FaceBounds) -> [simd_float3] {
        let amount = parameters.chinPoint * parameters.overallStrength
        guard amount > 0 else { return vertices }
        var result = vertices
        for i in vertices.indices {
            let v = vertices[i]
            let n = bounds.normalized(v)
            let vertWeight   = smoothstep(0.20, 0.48, -n.y)              // lower chin
            let centerWeight = 1.0 - smoothstep(0.0, 0.18, abs(n.x))    // center only
            let tipWeight    = vertWeight * centerWeight
            // Push chin tip downward
            result[i].y -= bounds.height * 0.028 * amount * tipWeight
            // Narrow chin sides toward center
            let sideWeight = smoothstep(0.06, 0.18, abs(n.x)) * vertWeight
            result[i].x -= sign(n.x) * bounds.width * 0.012 * amount * sideWeight
        }
        return result
    }

    // MARK: - Nose Slim (nose bridge narrowing)
    private func applyNoseSlim(_ vertices: [simd_float3], bounds: FaceBounds) -> [simd_float3] {
        let amount = parameters.noseSlim * parameters.overallStrength
        guard amount > 0 else { return vertices }
        var result = vertices
        for i in vertices.indices {
            let v = vertices[i]
            let n = bounds.normalized(v)
            // Skip center-line vertices: sign(0) = 1 in our impl which would move them asymmetrically
            guard abs(n.x) > 0.004, abs(n.x) < 0.11, n.y > -0.08, n.y < 0.14 else { continue }
            let horizontalWeight = 1.0 - smoothstep(0.02, 0.11, abs(n.x))
            let verticalWeight   = 1.0 - smoothstep(0.18, 0.30, abs(n.y - 0.03))
            let weight = horizontalWeight * verticalWeight
            result[i].x = v.x - sign(n.x) * bounds.width * 0.010 * amount * weight
        }
        return result
    }

    // MARK: - Nose Wing (alar base narrowing / 小鼻サイズ)
    private func applyNoseWing(_ vertices: [simd_float3], bounds: FaceBounds) -> [simd_float3] {
        let amount = parameters.noseWingSlim * parameters.overallStrength
        guard amount > 0 else { return vertices }
        var result = vertices
        for i in vertices.indices {
            let v = vertices[i]
            let n = bounds.normalized(v)
            guard abs(n.x) > 0.04, abs(n.x) < 0.15, n.y > -0.08, n.y < 0.06 else { continue }
            let horizontalWeight = smoothstep(0.04, 0.08, abs(n.x)) * (1.0 - smoothstep(0.11, 0.15, abs(n.x)))
            let verticalWeight   = 1.0 - smoothstep(0.10, 0.22, abs(n.y + 0.02))
            let weight = horizontalWeight * verticalWeight
            result[i].x = v.x - sign(n.x) * bounds.width * 0.018 * amount * weight
        }
        return result
    }

    // MARK: - Mouth Adjust
    private func applyMouthAdjust(_ vertices: [simd_float3], bounds: FaceBounds) -> [simd_float3] {
        let amount = parameters.mouthAdjust * parameters.overallStrength
        guard amount > 0 else { return vertices }
        let verticalCenter: Float = -0.18
        let radiusX: Float = 0.22
        let radiusY: Float = 0.10
        var result = vertices
        for i in vertices.indices {
            let v = vertices[i]
            let n = bounds.normalized(v)
            guard abs(n.x) < 0.20, n.y > -0.28, n.y < -0.08 else { continue }
            let dx = n.x
            let dy = n.y - verticalCenter
            let ed = sqrt((dx / radiusX) * (dx / radiusX) + (dy / radiusY) * (dy / radiusY))
            guard ed < 1.0 else { continue }
            let weight = 1.0 - smoothstep(0.0, 1.0, ed)
            result[i].x = v.x + sign(n.x) * bounds.width * 0.010 * amount * weight
        }
        return result
    }
}
