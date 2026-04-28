import simd

// Phase 1: region detection is handled via FaceBounds normalized coordinates.
// Phase 2: use stable ARKit vertex indices for precise landmark anchoring.
enum FaceRegion {
    case leftCheek, rightCheek, jaw, leftEye, rightEye, nose, mouth, forehead, chin
}

struct FaceLandmarkMapper {
    // Approximate eye centers in normalized face coordinates (nx, ny).
    // ARKit face space: x+ = subject's right, y+ = up.
    // These may require sign adjustment on device depending on camera orientation.
    static let leftEyeCenter  = simd_float2(-0.16,  0.12)
    static let rightEyeCenter = simd_float2( 0.16,  0.12)

    static func region(for normalized: simd_float2) -> FaceRegion {
        let nx = normalized.x, ny = normalized.y
        if ny > 0.30 { return .forehead }
        if ny < -0.35 { return .chin }
        if abs(nx) > 0.28 {
            return nx < 0 ? .leftCheek : .rightCheek
        }
        if ny > 0.05 && abs(nx) < 0.20 { return .nose }
        if ny < -0.10 && abs(nx) < 0.22 { return .mouth }
        if ny > 0.02 && abs(nx) > 0.12 {
            return nx < 0 ? .leftEye : .rightEye
        }
        return .jaw
    }
}
