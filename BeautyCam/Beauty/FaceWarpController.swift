import simd

final class FaceWarpController {
    var parameters: BeautyParameters = .default
    private let smoother = TemporalSmoother()

    func warp(vertices: [simd_float3]) -> [simd_float3] {
        guard !vertices.isEmpty else { return vertices }
        let bounds = FaceBounds(vertices: vertices)
        var result = vertices
        result = applyFaceSlim(result, bounds: bounds)
        result = applyJawSharpness(result, bounds: bounds)
        result = applyEyeScale(result, bounds: bounds)
        result = applyNoseSlim(result, bounds: bounds)
        result = applyMouthAdjust(result, bounds: bounds)
        result = smoother.smooth(result)
        return result
    }

    func reset() {
        smoother.reset()
    }

    // MARK: - Face Slim
    private func applyFaceSlim(_ vertices: [simd_float3], bounds: FaceBounds) -> [simd_float3] {
        let amount = parameters.faceSlim * parameters.overallStrength
        guard amount > 0 else { return vertices }
        var result = vertices
        for i in vertices.indices {
            let v = vertices[i]
            let n = bounds.normalized(v)
            let sideDistance = abs(n.x)
            let regionWeight   = smoothstep(0.22, 0.50, sideDistance)
            let verticalWeight = 1.0 - smoothstep(0.20, 0.42, abs(n.y))
            let weight = regionWeight * verticalWeight
            let side = sign(n.x)
            result[i].x = v.x - side * bounds.width * 0.045 * amount * weight
        }
        return result
    }

    // MARK: - Jaw Sharpness
    private func applyJawSharpness(_ vertices: [simd_float3], bounds: FaceBounds) -> [simd_float3] {
        let amount = parameters.jawSharpness * parameters.overallStrength
        guard amount > 0 else { return vertices }
        var result = vertices
        for i in vertices.indices {
            let v = vertices[i]
            let n = bounds.normalized(v)
            let lowerWeight = smoothstep(0.10, 0.45, -n.y)
            let sideWeight  = smoothstep(0.08, 0.32, abs(n.x))
            let weight = lowerWeight * sideWeight
            result[i].x = v.x - sign(n.x) * bounds.width * 0.035 * amount * weight
            // Subtle chin tip push
            let chinWeight = smoothstep(0.25, 0.50, -n.y) * (1.0 - smoothstep(0.05, 0.18, abs(n.x)))
            result[i].y -= bounds.height * 0.015 * amount * chinWeight
        }
        return result
    }

    // MARK: - Eye Scale
    private func applyEyeScale(_ vertices: [simd_float3], bounds: FaceBounds) -> [simd_float3] {
        let amount = parameters.eyeScale * parameters.overallStrength
        guard amount > 0 else { return vertices }
        let eyes: [simd_float2] = [FaceLandmarkMapper.leftEyeCenter, FaceLandmarkMapper.rightEyeCenter]
        let radiusX: Float = 0.13
        let radiusY: Float = 0.10
        var result = vertices
        for i in vertices.indices {
            let v = vertices[i]
            let n = bounds.normalized(v)
            for eyeCenter in eyes {
                let dx = n.x - eyeCenter.x
                let dy = n.y - eyeCenter.y
                let ed = sqrt((dx / radiusX) * (dx / radiusX) + (dy / radiusY) * (dy / radiusY))
                guard ed < 1.0 else { continue }
                let weight = 1.0 - smoothstep(0.0, 1.0, ed)
                let scaleX = 1.0 + 0.12 * amount * weight
                let scaleY = 1.0 + 0.20 * amount * weight
                let newNx = eyeCenter.x + dx * scaleX
                let newNy = eyeCenter.y + dy * scaleY
                result[i].x = bounds.centerX + newNx * bounds.width
                result[i].y = bounds.centerY + newNy * bounds.height
            }
        }
        return result
    }

    // MARK: - Nose Slim
    private func applyNoseSlim(_ vertices: [simd_float3], bounds: FaceBounds) -> [simd_float3] {
        let amount = parameters.noseSlim * parameters.overallStrength
        guard amount > 0 else { return vertices }
        var result = vertices
        for i in vertices.indices {
            let v = vertices[i]
            let n = bounds.normalized(v)
            guard abs(n.x) < 0.12, n.y > -0.10, n.y < 0.18 else { continue }
            let horizontalWeight = 1.0 - smoothstep(0.02, 0.13, abs(n.x))
            let verticalWeight   = 1.0 - smoothstep(0.20, 0.35, abs(n.y - 0.02))
            let weight = horizontalWeight * verticalWeight
            result[i].x = v.x - sign(n.x) * bounds.width * 0.025 * amount * weight
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
