import AVFoundation
import CoreGraphics
import CoreImage
import MetalKit
import MediaPipeTasksVision
import PhotosUI
import SwiftUI
import UIKit

struct BeautyParams: Equatable {
    var faceSlim: CGFloat
    var jawSlim: CGFloat
    var eyeScale: CGFloat
    var noseSlim: CGFloat
    var skinSmooth: CGFloat
    var skinBright: CGFloat
    var skinTone: CGFloat

    static let standard = BeautyParams(
        faceSlim: 0.50,
        jawSlim: 0.50,
        eyeScale: 0.50,
        noseSlim: 0.50,
        skinSmooth: 0.50,
        skinBright: 0.50,
        skinTone: 0.50
    )

    func mappedForEngine() -> BeautyParams {
        func eased(_ x: CGFloat) -> CGFloat {
            let v = x.clamped(to: 0...1)
            return 1 - pow(1 - v, 1.6)
        }

        return BeautyParams(
            faceSlim: eased(faceSlim),
            jawSlim: eased(jawSlim),
            eyeScale: 0.35 * eased(eyeScale),
            noseSlim: 0.30 * eased(noseSlim),
            skinSmooth: eased(skinSmooth),
            skinBright: eased(skinBright),
            skinTone: eased(skinTone)
        )
    }

    static func smoothed(previous: BeautyParams, current: BeautyParams, alpha: CGFloat) -> BeautyParams {
        BeautyParams(
            faceSlim: previous.faceSlim * alpha + current.faceSlim * (1 - alpha),
            jawSlim: previous.jawSlim * alpha + current.jawSlim * (1 - alpha),
            eyeScale: previous.eyeScale * alpha + current.eyeScale * (1 - alpha),
            noseSlim: previous.noseSlim * alpha + current.noseSlim * (1 - alpha),
            skinSmooth: previous.skinSmooth * alpha + current.skinSmooth * (1 - alpha),
            skinBright: previous.skinBright * alpha + current.skinBright * (1 - alpha),
            skinTone: previous.skinTone * alpha + current.skinTone * (1 - alpha)
        )
    }
}

struct FaceMetrics {
    var yawDegrees: CGFloat
    var pitchDegrees: CGFloat
    var rollDegrees: CGFloat
    var mouthOpen: CGFloat
    var smile: CGFloat
    var eyeOpen: CGFloat
}

private struct FaceTrackingState {
    var landmarks: [CGPoint]
    var metrics: FaceMetrics
    var smoothingAlpha: CGFloat
}

private final class FaceLandmarkEngine {
    private var landmarker: FaceLandmarker?
    private var previousLandmarks: [CGPoint] = []
    private var previousParams: BeautyParams?

    init() {
        landmarker = Self.makeLandmarker()
    }

    func process(pixelBuffer: CVPixelBuffer, timestampMs: Int, baseParams: BeautyParams) -> (tracking: FaceTrackingState?, params: BeautyParams) {
        guard let landmarker else {
            return (nil, baseParams)
        }

        guard let current = detectLandmarks(
            with: landmarker,
            pixelBuffer: pixelBuffer,
            timestampMs: timestampMs
        ) else {
            previousLandmarks = []
            previousParams = nil
            return (nil, baseParams)
        }
        let speed = averageMovement(current: current, previous: previousLandmarks)
        let alpha: CGFloat = speed > 0.018 ? 0.35 : 0.65

        let smoothed: [CGPoint]
        if previousLandmarks.count == current.count {
            smoothed = zip(previousLandmarks, current).map { prev, now in
                CGPoint(
                    x: prev.x * alpha + now.x * (1 - alpha),
                    y: prev.y * alpha + now.y * (1 - alpha)
                )
            }
        } else {
            smoothed = current
        }
        previousLandmarks = smoothed

        let metrics = computeFaceMetrics(landmarks: smoothed)
        var corrected = applyPoseAndExpression(baseParams: baseParams, metrics: metrics)
        if let previousParams {
            corrected = BeautyParams.smoothed(previous: previousParams, current: corrected, alpha: alpha)
        }
        previousParams = corrected

        return (
            FaceTrackingState(landmarks: smoothed, metrics: metrics, smoothingAlpha: alpha),
            corrected
        )
    }

    private static func makeLandmarker() -> FaceLandmarker? {
        guard let modelPath = Bundle.main.path(forResource: "face_landmarker", ofType: "task") else {
            print("[FaceLandmarker] face_landmarker.task is missing in app bundle.")
            return nil
        }

        let baseOptions = BaseOptions()
        baseOptions.modelAssetPath = modelPath

        let options = FaceLandmarkerOptions()
        options.baseOptions = baseOptions
        options.runningMode = .video
        options.numFaces = 1
        options.minFaceDetectionConfidence = 0.5
        options.minFacePresenceConfidence = 0.5
        options.minTrackingConfidence = 0.5

        do {
            return try FaceLandmarker(options: options)
        } catch {
            print("[FaceLandmarker] initialization failed: \(error)")
            return nil
        }
    }

    private func detectLandmarks(
        with landmarker: FaceLandmarker,
        pixelBuffer: CVPixelBuffer,
        timestampMs: Int
    ) -> [CGPoint]? {
        let orientations: [UIImage.Orientation] = [.leftMirrored, .rightMirrored, .up]

        for orientation in orientations {
            guard let image = try? MPImage(pixelBuffer: pixelBuffer, orientation: orientation) else {
                continue
            }

            guard let result = try? landmarker.detect(videoFrame: image, timestampInMilliseconds: timestampMs),
                  let firstFace = result.faceLandmarks.first,
                  !firstFace.isEmpty else {
                continue
            }

            return firstFace.map { CGPoint(x: CGFloat($0.x), y: CGFloat($0.y)) }
        }

        return nil
    }

    private func averageMovement(current: [CGPoint], previous: [CGPoint]) -> CGFloat {
        guard current.count == previous.count, !current.isEmpty else { return 0 }
        let total = zip(current, previous).reduce(CGFloat.zero) { partial, pair in
            partial + hypot(pair.0.x - pair.1.x, pair.0.y - pair.1.y)
        }
        return total / CGFloat(current.count)
    }

    private func computeFaceMetrics(landmarks: [CGPoint]) -> FaceMetrics {
        func point(_ index: Int) -> CGPoint? {
            guard index >= 0, index < landmarks.count else { return nil }
            return landmarks[index]
        }

        func distance(_ a: Int, _ b: Int) -> CGFloat {
            guard let p1 = point(a), let p2 = point(b) else { return 0 }
            return hypot(p1.x - p2.x, p1.y - p2.y)
        }

        let mouthOpen = distance(13, 14) / max(distance(78, 308), 0.0001)
        let leftEyeOpen = distance(159, 145) / max(distance(33, 133), 0.0001)
        let rightEyeOpen = distance(386, 374) / max(distance(362, 263), 0.0001)
        let eyeOpen = ((leftEyeOpen + rightEyeOpen) * 0.5).clamped(to: 0...1)

        var smile: CGFloat = 0
        if let leftCorner = point(61),
           let rightCorner = point(291),
           let upperLip = point(13),
           let lowerLip = point(14),
           let forehead = point(10),
           let chin = point(152) {
            let lipCenterY = (upperLip.y + lowerLip.y) * 0.5
            let cornersY = (leftCorner.y + rightCorner.y) * 0.5
            let faceHeight = max(hypot(forehead.x - chin.x, forehead.y - chin.y), 0.0001)
            smile = ((lipCenterY - cornersY) / faceHeight * 6).clamped(to: 0...1)
        }

        var yawDegrees: CGFloat = 0
        if let left = point(234), let right = point(454), let nose = point(1) {
            let leftSpan = abs(nose.x - left.x)
            let rightSpan = abs(right.x - nose.x)
            let imbalance = abs(leftSpan - rightSpan) / max(leftSpan + rightSpan, 0.0001)
            yawDegrees = min(70, imbalance * 140)
        }

        var pitchDegrees: CGFloat = 0
        if let leftEyeTop = point(159),
           let rightEyeTop = point(386),
           let upperLip = point(13),
           let lowerLip = point(14),
           let nose = point(1) {
            let eyeCenterY = (leftEyeTop.y + rightEyeTop.y) * 0.5
            let mouthCenterY = (upperLip.y + lowerLip.y) * 0.5
            let ratio = (nose.y - eyeCenterY) / max(mouthCenterY - eyeCenterY, 0.0001)
            pitchDegrees = min(60, abs(ratio - 0.5) * 120)
        }

        var rollDegrees: CGFloat = 0
        if let leftEye = point(33), let rightEye = point(263) {
            rollDegrees = atan2(rightEye.y - leftEye.y, rightEye.x - leftEye.x) * 180.0 / .pi
        }

        return FaceMetrics(
            yawDegrees: yawDegrees,
            pitchDegrees: pitchDegrees,
            rollDegrees: rollDegrees,
            mouthOpen: mouthOpen,
            smile: smile,
            eyeOpen: eyeOpen
        )
    }

    private func applyPoseAndExpression(baseParams: BeautyParams, metrics: FaceMetrics) -> BeautyParams {
        let yawFactor = 1 - min(abs(metrics.yawDegrees) / 35, 1)
        let pitchFactor = 1 - min(abs(metrics.pitchDegrees) / 30, 1)
        let poseFactor = max(0.2, yawFactor * pitchFactor)

        var corrected = baseParams
        corrected.faceSlim *= poseFactor
        corrected.jawSlim *= poseFactor
        corrected.eyeScale *= poseFactor
        corrected.noseSlim *= poseFactor

        if metrics.mouthOpen > 0.35 {
            corrected.jawSlim *= 0.5
        }
        if metrics.smile > 0.4 {
            corrected.faceSlim *= 0.75
        }
        if metrics.eyeOpen < 0.3 {
            corrected.eyeScale *= 0.2
        }

        return corrected
    }
}

struct FaceLineSlimInput {
    let image: CIImage
    let landmarks: [CGPoint]
    let imageSize: CGSize
    let strength: Float
    let yaw: Float
    let pitch: Float
    let roll: Float
}

struct FaceLineSlimOutput {
    let image: CIImage
}

struct MLSControlPoint {
    let source: CGPoint
    let target: CGPoint
    let weightScale: Float
}

struct FaceGeometry {
    let landmarks: [CGPoint]
    let imageSize: CGSize
    let faceCenter: CGPoint
    let faceWidth: CGFloat
    let faceHeight: CGFloat
    let contourPoints: [CGPoint]

    init(landmarks: [CGPoint], imageSize: CGSize) {
        self.landmarks = landmarks
        self.imageSize = imageSize

        let contourIndices = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
        self.contourPoints = contourIndices.compactMap { landmarks[safe: $0] }

        let stableCenterIndices = [1, 2, 4, 6, 10, 152, 168]
        let stablePoints = stableCenterIndices.compactMap { landmarks[safe: $0] }
        if stablePoints.isEmpty {
            self.faceCenter = CGPoint(x: imageSize.width * 0.5, y: imageSize.height * 0.5)
        } else {
            let sum = stablePoints.reduce(CGPoint.zero) { partial, point in
                CGPoint(x: partial.x + point.x, y: partial.y + point.y)
            }
            self.faceCenter = CGPoint(
                x: sum.x / CGFloat(stablePoints.count),
                y: sum.y / CGFloat(stablePoints.count)
            )
        }

        let minX = contourPoints.map(\ .x).min() ?? 0
        let maxX = contourPoints.map(\ .x).max() ?? imageSize.width
        let minY = contourPoints.map(\ .y).min() ?? 0
        let maxY = contourPoints.map(\ .y).max() ?? imageSize.height

        self.faceWidth = max(1, maxX - minX)
        self.faceHeight = max(1, maxY - minY)
    }
}

final class LandmarkSmoother {
    private var previous: [CGPoint]?

    func smooth(current: [CGPoint], motionSpeed: Float) -> [CGPoint] {
        let alpha: CGFloat = motionSpeed > 0.08 ? 0.35 : 0.65

        guard let previous, previous.count == current.count else {
            self.previous = current
            return current
        }

        let result = zip(previous, current).map { prev, cur in
            CGPoint(
                x: prev.x * alpha + cur.x * (1 - alpha),
                y: prev.y * alpha + cur.y * (1 - alpha)
            )
        }

        self.previous = result
        return result
    }
}

final class MLSRigidWarper {

    func warp(point v: CGPoint, controls: [MLSControlPoint], alpha: CGFloat) -> CGPoint {
        guard !controls.isEmpty else { return v }

        let epsilon: CGFloat = 1e-4
        var weightSum: CGFloat = 0
        var pStar = CGPoint.zero
        var qStar = CGPoint.zero

        var localWeights: [CGFloat] = []
        localWeights.reserveCapacity(controls.count)

        for control in controls {
            let dx = v.x - control.source.x
            let dy = v.y - control.source.y
            let dist = max(epsilon, hypot(dx, dy))
            let weight = CGFloat(control.weightScale) / pow(dist, 2 * alpha)
            localWeights.append(weight)
            weightSum += weight
            pStar.x += control.source.x * weight
            pStar.y += control.source.y * weight
            qStar.x += control.target.x * weight
            qStar.y += control.target.y * weight
        }

        guard weightSum > epsilon else { return v }

        pStar.x /= weightSum
        pStar.y /= weightSum
        qStar.x /= weightSum
        qStar.y /= weightSum

        let vHat = CGPoint(x: v.x - pStar.x, y: v.y - pStar.y)

        var a: CGFloat = 0
        var b: CGFloat = 0
        var mu: CGFloat = 0

        for (idx, control) in controls.enumerated() {
            let w = localWeights[idx]
            let pHat = CGPoint(x: control.source.x - pStar.x, y: control.source.y - pStar.y)
            let qHat = CGPoint(x: control.target.x - qStar.x, y: control.target.y - qStar.y)

            a += w * (pHat.x * qHat.x + pHat.y * qHat.y)
            b += w * (pHat.x * qHat.y - pHat.y * qHat.x)
            mu += w * (pHat.x * pHat.x + pHat.y * pHat.y)
        }

        guard mu > epsilon else { return v }

        let mapped = CGPoint(
            x: (a * vHat.x - b * vHat.y) / mu + qStar.x,
            y: (b * vHat.x + a * vHat.y) / mu + qStar.y
        )

        return mapped
    }

    func warpImage(_ image: CIImage, controls: [MLSControlPoint], imageSize: CGSize) -> CIImage {
        guard !controls.isEmpty else { return image }

        let gridCols = 32
        let gridRows = 48
        let maxDisp = max(28.0, min(imageSize.width, imageSize.height) * 0.18)

        var bytes = [UInt8](repeating: 0, count: gridCols * gridRows * 4)

        for row in 0 ..< gridRows {
            for col in 0 ..< gridCols {
                let u = CGFloat(col) / CGFloat(gridCols - 1)
                let v = CGFloat(row) / CGFloat(gridRows - 1)

                let sourcePoint = CGPoint(x: u * imageSize.width, y: v * imageSize.height)
                let warpedPoint = warp(point: sourcePoint, controls: controls, alpha: 1.0)
                let dx = (warpedPoint.x - sourcePoint.x).clamped(to: -maxDisp...maxDisp)
                let dy = (warpedPoint.y - sourcePoint.y).clamped(to: -maxDisp...maxDisp)

                let r = UInt8(((dx / maxDisp) * 0.5 + 0.5) * 255.0)
                let g = UInt8(((dy / maxDisp) * 0.5 + 0.5) * 255.0)

                let index = (row * gridCols + col) * 4
                bytes[index + 0] = r
                bytes[index + 1] = g
                bytes[index + 2] = 128
                bytes[index + 3] = 255
            }
        }

        let data = Data(bytes)
        let gridImage = CIImage(
            bitmapData: data,
            bytesPerRow: gridCols * 4,
            size: CGSize(width: gridCols, height: gridRows),
            format: .RGBA8,
            colorSpace: CGColorSpaceCreateDeviceRGB()
        )

        let scaledMap = gridImage
            .transformed(by: CGAffineTransform(scaleX: imageSize.width / CGFloat(gridCols), y: imageSize.height / CGFloat(gridRows)))
            .cropped(to: CGRect(origin: .zero, size: imageSize))
            .clampedToExtent()
            .applyingFilter("CIGaussianBlur", parameters: [kCIInputRadiusKey: 0.8])
            .cropped(to: CGRect(origin: .zero, size: imageSize))

        return image.applyingFilter(
            "CIDisplacementDistortion",
            parameters: [
                "inputDisplacementImage": scaledMap,
                kCIInputScaleKey: maxDisp
            ]
        ).cropped(to: image.extent)
    }
}

final class FaceLineSlimEffect {
    private let smoother = LandmarkSmoother()
    private let warper = MLSRigidWarper()
    private var previousRawLandmarks: [CGPoint]?

    func apply(input: FaceLineSlimInput) -> FaceLineSlimOutput {
        let motionSpeed = estimateMotionSpeed(input.landmarks)

        let smoothedLandmarks = smoother.smooth(
            current: input.landmarks,
            motionSpeed: motionSpeed
        )

        let geometry = FaceGeometry(
            landmarks: smoothedLandmarks,
            imageSize: input.imageSize
        )

        // Pose correction is already applied in FaceLandmarkEngine.
        let effectiveStrength = input.strength.clamped(to: 0...1)

        let controls = makeFaceLineControls(
            geometry: geometry,
            strength: effectiveStrength
        )

        let warpedImage = warper.warpImage(
            input.image,
            controls: controls,
            imageSize: input.imageSize
        )

        let mask = makeFaceMask(
            geometry: geometry,
            imageSize: input.imageSize
        )

        let composited = composite(
            original: input.image,
            warped: warpedImage,
            mask: mask
        )

        return FaceLineSlimOutput(image: composited)
    }

    private func estimateMotionSpeed(_ landmarks: [CGPoint]) -> Float {
        defer { previousRawLandmarks = landmarks }

        guard let previousRawLandmarks,
              previousRawLandmarks.count == landmarks.count,
              !landmarks.isEmpty else {
            return 0
        }

        let total = zip(previousRawLandmarks, landmarks).reduce(CGFloat.zero) { partial, pair in
            partial + hypot(pair.0.x - pair.1.x, pair.0.y - pair.1.y)
        }

        return Float(total / CGFloat(landmarks.count))
    }

    private func poseFactor(yaw: Float, pitch: Float, roll: Float) -> Float {
        let yawFactor = 1.0 - min(abs(yaw) / 35.0, 1.0)
        let pitchFactor = 1.0 - min(abs(pitch) / 30.0, 1.0)
        let rollFactor = 1.0 - min(abs(roll) / 45.0, 1.0)
        return max(0.25, yawFactor * pitchFactor * rollFactor)
    }

    private func makeFaceLineControls(geometry: FaceGeometry, strength: Float) -> [MLSControlPoint] {
        let cheekWeight: CGFloat = 0.35
        let jawWeight: CGFloat = 1.00
        let chinSideWeight: CGFloat = 0.70
        let chinTipWeight: CGFloat = 0.35

        let effectiveStrength = CGFloat(strength.clamped(to: 0...1))
        let baseRatio = (0.08 * effectiveStrength) + (0.08 * effectiveStrength * effectiveStrength)
        let baseAmount = min(geometry.faceWidth * 0.18, geometry.faceWidth * baseRatio)

        let leftCheek = [234, 93, 132]
        let leftJaw = [58, 172, 136, 150]
        let chin = [149, 148, 152, 377, 400]
        let rightJaw = [378, 365, 397, 288]
        let rightCheek = [361, 323, 454]

        var controls: [MLSControlPoint] = []

        func addHorizontalControls(indices: [Int], isLeft: Bool, groupWeight: CGFloat) {
            for idx in indices {
                guard let source = geometry.landmarks[safe: idx] else { continue }
                let direction: CGFloat = isLeft ? 1 : -1
                let target = CGPoint(
                    x: source.x + direction * baseAmount * groupWeight,
                    y: source.y
                )
                controls.append(MLSControlPoint(source: source, target: target, weightScale: 1.0))
            }
        }

        addHorizontalControls(indices: leftCheek, isLeft: true, groupWeight: cheekWeight)
        addHorizontalControls(indices: leftJaw, isLeft: true, groupWeight: jawWeight)
        addHorizontalControls(indices: rightJaw, isLeft: false, groupWeight: jawWeight)
        addHorizontalControls(indices: rightCheek, isLeft: false, groupWeight: cheekWeight)

        for idx in chin {
            guard let source = geometry.landmarks[safe: idx] else { continue }
            let horizontal = (geometry.faceCenter.x - source.x) * 0.10 * effectiveStrength
            let vertical = min(geometry.faceHeight * 0.012 * effectiveStrength, geometry.faceHeight * 0.018)
            let groupWeight = idx == 152 ? chinTipWeight : chinSideWeight
            let target = CGPoint(
                x: source.x + horizontal * groupWeight,
                y: source.y + vertical * groupWeight
            )
            controls.append(MLSControlPoint(source: source, target: target, weightScale: 1.0))
        }

        let anchorIndices = [6, 1, 10, 33, 263, 13]
        for idx in anchorIndices {
            guard let point = geometry.landmarks[safe: idx] else { continue }
            controls.append(MLSControlPoint(source: point, target: point, weightScale: 0.35))
        }
        controls.append(MLSControlPoint(source: geometry.faceCenter, target: geometry.faceCenter, weightScale: 0.45))

        let contour = geometry.contourPoints
        let minX = contour.map(\ .x).min() ?? 0
        let maxX = contour.map(\ .x).max() ?? geometry.imageSize.width
        let minY = contour.map(\ .y).min() ?? 0
        let maxY = contour.map(\ .y).max() ?? geometry.imageSize.height

        let padX = geometry.faceWidth * 0.20
        let padY = geometry.faceHeight * 0.24

        let bgAnchors = [
            CGPoint(x: minX - padX, y: minY - padY),
            CGPoint(x: maxX + padX, y: minY - padY),
            CGPoint(x: minX - padX, y: maxY + padY),
            CGPoint(x: maxX + padX, y: maxY + padY),
            CGPoint(x: minX - padX, y: geometry.faceCenter.y),
            CGPoint(x: maxX + padX, y: geometry.faceCenter.y),
            CGPoint(x: geometry.faceCenter.x, y: minY - padY),
            CGPoint(x: geometry.faceCenter.x, y: maxY + padY)
        ]

        for anchor in bgAnchors {
            let clamped = CGPoint(
                x: anchor.x.clamped(to: 0...geometry.imageSize.width),
                y: anchor.y.clamped(to: 0...geometry.imageSize.height)
            )
            controls.append(MLSControlPoint(source: clamped, target: clamped, weightScale: 0.15))
        }

        return controls
    }

    private func makeFaceMask(geometry: FaceGeometry, imageSize: CGSize) -> CIImage {
        guard !geometry.contourPoints.isEmpty else {
            return CIImage(color: .black).cropped(to: CGRect(origin: .zero, size: imageSize))
        }

        let minX = geometry.contourPoints.map(\ .x).min() ?? 0
        let maxX = geometry.contourPoints.map(\ .x).max() ?? imageSize.width
        let minY = geometry.contourPoints.map(\ .y).min() ?? 0
        let maxY = geometry.contourPoints.map(\ .y).max() ?? imageSize.height

        let maskExpand = geometry.faceWidth * 0.10
        let expandedWidth = (maxX - minX) + maskExpand * 2
        let expandedHeight = (maxY - minY) + maskExpand * 2

        let radius = max(expandedWidth, expandedHeight) * 0.55
        let maskBlur = max(1.0, geometry.faceWidth * 0.05)

        let base = radialMask(
            center: geometry.faceCenter,
            radius: radius,
            softness: 0.42,
            extent: CGRect(origin: .zero, size: imageSize)
        )

        return base
            .clampedToExtent()
            .applyingFilter("CIGaussianBlur", parameters: [kCIInputRadiusKey: maskBlur])
            .cropped(to: CGRect(origin: .zero, size: imageSize))
    }

    private func radialMask(center: CGPoint, radius: CGFloat, softness: CGFloat, extent: CGRect) -> CIImage {
        let innerRadius = max(1, radius * (1 - softness))
        return CIFilter(
            name: "CIRadialGradient",
            parameters: [
                "inputCenter": CIVector(cgPoint: center),
                "inputRadius0": innerRadius,
                "inputRadius1": radius,
                "inputColor0": CIColor(red: 1, green: 1, blue: 1, alpha: 1),
                "inputColor1": CIColor(red: 0, green: 0, blue: 0, alpha: 0)
            ]
        )?.outputImage?.cropped(to: extent) ?? CIImage(color: .black).cropped(to: extent)
    }

    private func composite(original: CIImage, warped: CIImage, mask: CIImage) -> CIImage {
        let boostedMask = mask.applyingFilter(
            "CIColorMatrix",
            parameters: [
                "inputRVector": CIVector(x: 1.35, y: 0, z: 0, w: 0),
                "inputGVector": CIVector(x: 0, y: 1.35, z: 0, w: 0),
                "inputBVector": CIVector(x: 0, y: 0, z: 1.35, w: 0),
                "inputAVector": CIVector(x: 0, y: 0, z: 0, w: 1.35)
            ]
        )

        let blended = warped.applyingFilter(
            "CIBlendWithMask",
            parameters: [
                kCIInputBackgroundImageKey: original,
                kCIInputMaskImageKey: boostedMask
            ]
        )

        // Slightly bias towards warped image so that subtle deformation remains visible.
        return blended.applyingFilter(
            "CIColorControls",
            parameters: [kCIInputSaturationKey: 1.02]
        )
    }
}

private final class BeautyPipeline {
    private let faceLineSlimEffect = FaceLineSlimEffect()

    func process(image: CIImage, tracking: FaceTrackingState?, params: BeautyParams, debugOverlay: Bool) -> CIImage {
        guard let tracking else { return image }

        let imageSize = image.extent.size
        let points = tracking.landmarks.map { normalized -> CGPoint in
            CGPoint(
                x: image.extent.minX + normalized.x * image.extent.width,
                y: image.extent.minY + (1 - normalized.y) * image.extent.height
            )
        }

        // Keep full slider dynamic range; do not floor small values.
        let rawStrength = (params.faceSlim * 0.65 + params.jawSlim * 0.95).clamped(to: 0...1)
        let strength = Float(rawStrength)

        let slimmed = faceLineSlimEffect.apply(
            input: FaceLineSlimInput(
                image: image,
                landmarks: points,
                imageSize: imageSize,
                strength: strength,
                yaw: Float(tracking.metrics.yawDegrees),
                pitch: Float(tracking.metrics.pitchDegrees),
                roll: Float(tracking.metrics.rollDegrees)
            )
        ).image

        let reshaped = applyFeatureDistortionPass(image: slimmed, points: points, params: params)

        let faceMask = makeSkinMask(extent: image.extent, points: points)
        let smoothed = applySkinPass(image: reshaped, mask: faceMask, params: params)
        var colored = applyColorPass(image: smoothed, mask: faceMask, params: params)

        if debugOverlay {
            colored = drawDebugOverlay(on: colored, points: points)
        }

        return colored
    }

    private func makeSkinMask(extent: CGRect, points: [CGPoint]) -> CIImage {
        let faceOval = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]

        let ovalPoints = faceOval.compactMap { points[safe: $0] }
        guard !ovalPoints.isEmpty else {
            return CIImage(color: .black).cropped(to: extent)
        }

        let bounds = ovalPoints.reduce(into: CGRect.null) { partial, point in
            partial = partial.union(CGRect(x: point.x, y: point.y, width: 1, height: 1))
        }

        let center = CGPoint(x: bounds.midX, y: bounds.midY)

        let upperFaceMask = radialMask(
            center: CGPoint(x: center.x, y: center.y - bounds.height * 0.14),
            radius: max(bounds.width * 0.33, bounds.height * 0.26),
            softness: 0.42,
            extent: extent
        )
        let lowerFaceMask = radialMask(
            center: CGPoint(x: center.x, y: center.y + bounds.height * 0.12),
            radius: max(bounds.width * 0.30, bounds.height * 0.24),
            softness: 0.42,
            extent: extent
        )

        var baseMask = upperFaceMask.applyingFilter(
            "CIMaximumCompositing",
            parameters: [kCIInputBackgroundImageKey: lowerFaceMask]
        ).cropped(to: extent)

        let leftEyeHole = radialMask(
            center: averagePoint(points: points, indices: [33, 133, 159, 145]) ?? center,
            radius: max(16, bounds.width * 0.055),
            softness: 0.55,
            extent: extent
        )

        let rightEyeHole = radialMask(
            center: averagePoint(points: points, indices: [362, 263, 386, 374]) ?? center,
            radius: max(16, bounds.width * 0.055),
            softness: 0.55,
            extent: extent
        )

        let lipHole = radialMask(
            center: averagePoint(points: points, indices: [13, 14, 61, 291]) ?? center,
            radius: max(18, bounds.width * 0.075),
            softness: 0.50,
            extent: extent
        )

        baseMask = subtractHole(from: baseMask, hole: leftEyeHole, extent: extent)
        baseMask = subtractHole(from: baseMask, hole: rightEyeHole, extent: extent)
        baseMask = subtractHole(from: baseMask, hole: lipHole, extent: extent)

        return baseMask
    }

    private func applySkinPass(image: CIImage, mask: CIImage, params: BeautyParams) -> CIImage {
        let radius = 2.0 + params.skinSmooth * 20.0
        let smooth = image
            .clampedToExtent()
            .applyingFilter("CIGaussianBlur", parameters: [kCIInputRadiusKey: radius])
            .cropped(to: image.extent)

        let smoothMask = mask.applyingFilter(
            "CIColorMatrix",
            parameters: [
                "inputRVector": CIVector(x: params.skinSmooth, y: 0, z: 0, w: 0),
                "inputGVector": CIVector(x: 0, y: params.skinSmooth, z: 0, w: 0),
                "inputBVector": CIVector(x: 0, y: 0, z: params.skinSmooth, w: 0),
                "inputAVector": CIVector(x: 0, y: 0, z: 0, w: params.skinSmooth)
            ]
        )

        return smooth.applyingFilter(
            "CIBlendWithMask",
            parameters: [
                kCIInputBackgroundImageKey: image,
                kCIInputMaskImageKey: smoothMask
            ]
        )
    }

    private func applyColorPass(image: CIImage, mask: CIImage, params: BeautyParams) -> CIImage {
        let bright = image.applyingFilter(
            "CIColorControls",
            parameters: [
                kCIInputBrightnessKey: params.skinBright * 0.45,
                kCIInputSaturationKey: 1.0 + params.skinBright * 0.18
            ]
        )

        let whitened = bright.applyingFilter(
            "CIBlendWithMask",
            parameters: [
                kCIInputBackgroundImageKey: image,
                kCIInputMaskImageKey: mask
            ]
        )

        let rednessReduced = whitened.applyingFilter(
            "CIColorMatrix",
            parameters: [
                "inputRVector": CIVector(x: max(0.45, 1 - params.skinTone * 0.55), y: 0, z: 0, w: 0),
                "inputGVector": CIVector(x: 0, y: 1, z: 0, w: 0),
                "inputBVector": CIVector(x: 0, y: 0, z: 1, w: 0),
                "inputAVector": CIVector(x: 0, y: 0, z: 0, w: 1)
            ]
        )

        let toneMask = mask.applyingFilter(
            "CIColorMatrix",
            parameters: [
                "inputRVector": CIVector(x: params.skinTone, y: 0, z: 0, w: 0),
                "inputGVector": CIVector(x: 0, y: params.skinTone, z: 0, w: 0),
                "inputBVector": CIVector(x: 0, y: 0, z: params.skinTone, w: 0),
                "inputAVector": CIVector(x: 0, y: 0, z: 0, w: params.skinTone)
            ]
        )

        return rednessReduced.applyingFilter(
            "CIBlendWithMask",
            parameters: [
                kCIInputBackgroundImageKey: whitened,
                kCIInputMaskImageKey: toneMask
            ]
        )
    }

    private func applyFeatureDistortionPass(image: CIImage, points: [CGPoint], params: BeautyParams) -> CIImage {
        var result = image

        let leftCheekCenter = averagePoint(points: points, indices: [234, 132, 93])
        let rightCheekCenter = averagePoint(points: points, indices: [454, 361, 323])
        let leftJawCenter = averagePoint(points: points, indices: [58, 172, 136])
        let rightJawCenter = averagePoint(points: points, indices: [288, 397, 365])
        let leftEyeCenter = averagePoint(points: points, indices: [33, 133, 159, 145])
        let rightEyeCenter = averagePoint(points: points, indices: [362, 263, 386, 374])
        let noseCenter = points[safe: 1]

        let faceWidth = (points[safe: 454]?.x ?? 0) - (points[safe: 234]?.x ?? 0)
        let width = max(80, abs(faceWidth))

        let faceRadius = width * 0.29
        let jawRadius = width * 0.24
        let eyeRadius = width * 0.16
        let noseRadius = width * 0.14

        if let leftCheekCenter {
            result = result.applyingFilter(
                "CIBumpDistortion",
                parameters: [
                    kCIInputCenterKey: CIVector(cgPoint: leftCheekCenter),
                    kCIInputRadiusKey: faceRadius,
                    kCIInputScaleKey: -0.78 * params.faceSlim
                ]
            )
        }

        if let rightCheekCenter {
            result = result.applyingFilter(
                "CIBumpDistortion",
                parameters: [
                    kCIInputCenterKey: CIVector(cgPoint: rightCheekCenter),
                    kCIInputRadiusKey: faceRadius,
                    kCIInputScaleKey: -0.78 * params.faceSlim
                ]
            )
        }

        if let leftJawCenter {
            result = result.applyingFilter(
                "CIBumpDistortion",
                parameters: [
                    kCIInputCenterKey: CIVector(cgPoint: leftJawCenter),
                    kCIInputRadiusKey: jawRadius,
                    kCIInputScaleKey: -0.95 * params.jawSlim
                ]
            )
        }

        if let rightJawCenter {
            result = result.applyingFilter(
                "CIBumpDistortion",
                parameters: [
                    kCIInputCenterKey: CIVector(cgPoint: rightJawCenter),
                    kCIInputRadiusKey: jawRadius,
                    kCIInputScaleKey: -0.95 * params.jawSlim
                ]
            )
        }

        if let leftEyeCenter {
            result = result.applyingFilter(
                "CIBumpDistortion",
                parameters: [
                    kCIInputCenterKey: CIVector(cgPoint: leftEyeCenter),
                    kCIInputRadiusKey: eyeRadius,
                    kCIInputScaleKey: 1.35 * params.eyeScale
                ]
            )
        }

        if let rightEyeCenter {
            result = result.applyingFilter(
                "CIBumpDistortion",
                parameters: [
                    kCIInputCenterKey: CIVector(cgPoint: rightEyeCenter),
                    kCIInputRadiusKey: eyeRadius,
                    kCIInputScaleKey: 1.35 * params.eyeScale
                ]
            )
        }

        if let noseCenter {
            result = result.applyingFilter(
                "CIBumpDistortion",
                parameters: [
                    kCIInputCenterKey: CIVector(cgPoint: noseCenter),
                    kCIInputRadiusKey: noseRadius,
                    kCIInputScaleKey: -1.15 * params.noseSlim
                ]
            )
        }

        // Preserve background space by blending warped result only inside a soft face mask.
        let extent = image.extent
        let center = averagePoint(points: points, indices: [1, 2, 4, 10, 152])
            ?? CGPoint(x: extent.midX, y: extent.midY)
        let chin = points[safe: 152] ?? center

        let faceMask = radialMask(
            center: center,
            radius: width * 0.66,
            softness: 0.42,
            extent: extent
        )

        let jawMask = radialMask(
            center: CGPoint(x: chin.x, y: chin.y - width * 0.02),
            radius: width * 0.40,
            softness: 0.52,
            extent: extent
        )

        let leftContourMask = radialMask(
            center: leftJawCenter ?? CGPoint(x: center.x - width * 0.20, y: center.y + width * 0.10),
            radius: width * 0.30,
            softness: 0.56,
            extent: extent
        )

        let rightContourMask = radialMask(
            center: rightJawCenter ?? CGPoint(x: center.x + width * 0.20, y: center.y + width * 0.10),
            radius: width * 0.30,
            softness: 0.56,
            extent: extent
        )

        let mergedMask = faceMask
            .applyingFilter("CIMaximumCompositing", parameters: [kCIInputBackgroundImageKey: jawMask])
            .applyingFilter("CIMaximumCompositing", parameters: [kCIInputBackgroundImageKey: leftContourMask])
            .applyingFilter("CIMaximumCompositing", parameters: [kCIInputBackgroundImageKey: rightContourMask])
            .clampedToExtent()
            .applyingFilter("CIGaussianBlur", parameters: [kCIInputRadiusKey: width * 0.03])
            .cropped(to: extent)

        return result
            .applyingFilter(
                "CIBlendWithMask",
                parameters: [
                    kCIInputBackgroundImageKey: image,
                    kCIInputMaskImageKey: mergedMask
                ]
            )
            .cropped(to: extent)
    }

    private func drawDebugOverlay(on image: CIImage, points: [CGPoint]) -> CIImage {
        var result = image

        // Mesh confirmation overlay.
        for (index, p) in points.enumerated() where index % 4 == 0 {
            result = overlayDot(
                on: result,
                center: p,
                radius: 2.4,
                color: CIColor(red: 0.08, green: 0.78, blue: 1.0, alpha: 0.98)
            )
        }

        let contourIndices = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]

        for idx in contourIndices {
            guard let p = points[safe: idx] else { continue }
            result = overlayDot(
                on: result,
                center: p,
                radius: 2.4,
                color: CIColor(red: 0.15, green: 1.0, blue: 0.15, alpha: 0.95)
            )
        }

        if let center = averagePoint(points: points, indices: [1, 2, 4, 10, 152]) {
            result = overlayDot(
                on: result,
                center: center,
                radius: 4.2,
                color: CIColor(red: 1.0, green: 0.2, blue: 0.2, alpha: 0.95)
            )
        }

        return result
    }

    private func overlayDot(on image: CIImage, center: CGPoint, radius: CGFloat, color: CIColor) -> CIImage {
        let dot = CIFilter(
            name: "CIRadialGradient",
            parameters: [
                "inputCenter": CIVector(cgPoint: center),
                "inputRadius0": radius,
                "inputRadius1": radius + 0.01,
                "inputColor0": color,
                "inputColor1": CIColor(red: color.red, green: color.green, blue: color.blue, alpha: 0)
            ]
        )?.outputImage?.cropped(to: image.extent)

        guard let dot else { return image }

        return dot.applyingFilter(
            "CISourceOverCompositing",
            parameters: [kCIInputBackgroundImageKey: image]
        ).cropped(to: image.extent)
    }

    private func averagePoint(points: [CGPoint], indices: [Int]) -> CGPoint? {
        let valid = indices.compactMap { points[safe: $0] }
        guard !valid.isEmpty else { return nil }

        let sum = valid.reduce(CGPoint.zero) { partial, point in
            CGPoint(x: partial.x + point.x, y: partial.y + point.y)
        }

        return CGPoint(x: sum.x / CGFloat(valid.count), y: sum.y / CGFloat(valid.count))
    }

    private func radialMask(center: CGPoint, radius: CGFloat, softness: CGFloat, extent: CGRect) -> CIImage {
        let innerRadius = max(1, radius * (1 - softness))

        return CIFilter(
            name: "CIRadialGradient",
            parameters: [
                "inputCenter": CIVector(cgPoint: center),
                "inputRadius0": innerRadius,
                "inputRadius1": radius,
                "inputColor0": CIColor(red: 1, green: 1, blue: 1, alpha: 1),
                "inputColor1": CIColor(red: 0, green: 0, blue: 0, alpha: 0)
            ]
        )?.outputImage?.cropped(to: extent) ?? CIImage(color: .black).cropped(to: extent)
    }

    private func subtractHole(from baseMask: CIImage, hole: CIImage, extent: CGRect) -> CIImage {
        let invertedHole = hole.applyingFilter("CIColorInvert").cropped(to: extent)

        return invertedHole.applyingFilter(
            "CIMultiplyCompositing",
            parameters: [kCIInputBackgroundImageKey: baseMask]
        ).cropped(to: extent)
    }
}

private extension Array {
    subscript(safe index: Int) -> Element? {
        guard indices.contains(index) else { return nil }
        return self[index]
    }
}

private extension CGFloat {
    func clamped(to range: ClosedRange<CGFloat>) -> CGFloat {
        Swift.min(Swift.max(self, range.lowerBound), range.upperBound)
    }
}

private extension Float {
    func clamped(to range: ClosedRange<Float>) -> Float {
        Swift.min(Swift.max(self, range.lowerBound), range.upperBound)
    }
}

private extension UIImage {
    func normalizedUpOrientation() -> UIImage {
        guard imageOrientation != .up else { return self }
        let renderer = UIGraphicsImageRenderer(size: size)
        return renderer.image { _ in
            draw(in: CGRect(origin: .zero, size: size))
        }
    }

    func resizedIfNeeded(maxDimension: CGFloat) -> UIImage {
        let maxSide = max(size.width, size.height)
        guard maxSide > maxDimension, maxSide > 0 else { return self }

        let scale = maxDimension / maxSide
        let target = CGSize(width: size.width * scale, height: size.height * scale)
        let renderer = UIGraphicsImageRenderer(size: target)
        return renderer.image { _ in
            draw(in: CGRect(origin: .zero, size: target))
        }
    }
}

private enum EditorMode: String, CaseIterable, Identifiable {
    case live = "Live"
    case photo = "Photo"

    var id: String { rawValue }
}

struct ContentView: View {
    @State private var mode: EditorMode = .photo
    @State private var faceSlim: Double = 0.18
    @State private var jawSlim: Double = 0.16
    @State private var eyeScale: Double = 0.10
    @State private var noseSlim: Double = 0.06
    @State private var skinSmooth: Double = 0.35
    @State private var skinBright: Double = 0.10
    @State private var skinTone: Double = 0.08
    @State private var debugOverlay: Bool = false
    @State private var photoApplyToken: Int = 0

    private var currentParams: BeautyParams {
        BeautyParams(
            faceSlim: faceSlim,
            jawSlim: jawSlim,
            eyeScale: eyeScale,
            noseSlim: noseSlim,
            skinSmooth: skinSmooth,
            skinBright: skinBright,
            skinTone: skinTone
        )
    }

    var body: some View {
        ZStack(alignment: .bottom) {
            if mode == .live {
                CameraView(baseParams: currentParams, debugOverlay: debugOverlay)
                    .ignoresSafeArea()
            } else {
                PhotoBeautyView(baseParams: currentParams, debugOverlay: debugOverlay, applyToken: photoApplyToken)
                    .ignoresSafeArea()
            }

            VStack(spacing: 8) {
                Picker("Mode", selection: $mode) {
                    ForEach(EditorMode.allCases) { item in
                        Text(item.rawValue).tag(item)
                    }
                }
                .pickerStyle(.segmented)

                Toggle("Debug Overlay (Mesh)", isOn: $debugOverlay)
                    .font(.caption)
                    .tint(.green)

                ParameterSliderRow(title: "Face Slim", value: $faceSlim, range: 0...1.0) { editing in
                    if !editing, mode == .photo { photoApplyToken += 1 }
                }
                ParameterSliderRow(title: "Jaw Slim", value: $jawSlim, range: 0...1.0) { editing in
                    if !editing, mode == .photo { photoApplyToken += 1 }
                }
                ParameterSliderRow(title: "Eye Scale", value: $eyeScale, range: 0...1.0) { editing in
                    if !editing, mode == .photo { photoApplyToken += 1 }
                }
                ParameterSliderRow(title: "Nose Slim", value: $noseSlim, range: 0...1.0) { editing in
                    if !editing, mode == .photo { photoApplyToken += 1 }
                }
                ParameterSliderRow(title: "Skin Smooth", value: $skinSmooth, range: 0...1.0) { editing in
                    if !editing, mode == .photo { photoApplyToken += 1 }
                }
                ParameterSliderRow(title: "Skin Bright", value: $skinBright, range: 0...1.0) { editing in
                    if !editing, mode == .photo { photoApplyToken += 1 }
                }
                ParameterSliderRow(title: "Skin Tone", value: $skinTone, range: 0...1.0) { editing in
                    if !editing, mode == .photo { photoApplyToken += 1 }
                }
            }
            .padding(.horizontal, 16)
            .padding(.vertical, 14)
            .background(.black.opacity(0.45), in: RoundedRectangle(cornerRadius: 18, style: .continuous))
            .padding(.horizontal, 14)
            .padding(.bottom, 20)
        }
        .background(.black)
    }
}

private struct PhotoBeautyView: View {
    var baseParams: BeautyParams
    var debugOverlay: Bool
    var applyToken: Int

    @State private var selectedItem: PhotosPickerItem?
    @State private var sourceImage: UIImage?
    @State private var renderedImage: UIImage?
    @State private var tracking: FaceTrackingState?
    @State private var isProcessing = false
    @State private var renderRevision: Int = 0

    private let analyzer = StillFaceAnalyzer()
    private let pipeline = BeautyPipeline()
    private let ciContext = CIContext(options: [.cacheIntermediates: false])
    private let renderQueue = DispatchQueue(label: "BeautyCam.photo.render", qos: .userInitiated)

    var body: some View {
        ZStack {
            Color.black.ignoresSafeArea()

            VStack(spacing: 12) {
                HStack(spacing: 12) {
                    PhotosPicker(selection: $selectedItem, matching: .images, photoLibrary: .shared()) {
                        Text("写真を選択")
                            .font(.subheadline.weight(.semibold))
                            .foregroundStyle(.black)
                            .padding(.horizontal, 14)
                            .padding(.vertical, 10)
                            .background(.white, in: RoundedRectangle(cornerRadius: 10, style: .continuous))
                    }

                    Button("再適用") {
                        renderCurrentPhoto()
                    }
                    .font(.subheadline.weight(.semibold))
                    .foregroundStyle(.white)
                    .padding(.horizontal, 14)
                    .padding(.vertical, 10)
                    .background(.white.opacity(0.14), in: RoundedRectangle(cornerRadius: 10, style: .continuous))
                    .disabled(sourceImage == nil || tracking == nil)

                    if isProcessing {
                        ProgressView()
                            .tint(.white)
                    }
                }

                if let image = renderedImage ?? sourceImage {
                    Image(uiImage: image)
                        .resizable()
                        .scaledToFit()
                        .frame(maxWidth: .infinity, maxHeight: .infinity)
                        .background(Color.black)
                } else {
                    VStack(spacing: 12) {
                        Image(systemName: "photo")
                            .font(.system(size: 40, weight: .regular))
                            .foregroundStyle(.white.opacity(0.75))
                        Text("写真を選択すると同じエフェクトを静止画に適用します")
                            .font(.footnote)
                            .foregroundStyle(.white.opacity(0.75))
                    }
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
                }
            }
            .padding(.horizontal, 14)
            .padding(.top, 40)
            .padding(.bottom, 180)
        }
        .onChange(of: selectedItem) { _, item in
            guard let item else { return }
            loadPhoto(from: item)
        }
        .onChange(of: applyToken) { _, _ in
            renderCurrentPhoto()
        }
        .onChange(of: debugOverlay) { _, _ in
            renderCurrentPhoto()
        }
    }

    private func loadPhoto(from item: PhotosPickerItem) {
        Task {
            guard let data = try? await item.loadTransferable(type: Data.self),
                  let image = UIImage(data: data) else {
                return
            }

            // Normalize and downscale once to keep memory bounded in repeated re-renders.
            let normalized = image.normalizedUpOrientation()
            let prepared = normalized.resizedIfNeeded(maxDimension: 1536)

            sourceImage = prepared
            renderedImage = prepared
            tracking = analyzer.analyze(image: prepared)
            renderCurrentPhoto()
        }
    }

    private func renderCurrentPhoto() {
        guard let sourceImage, let tracking else {
            return
        }

        renderRevision += 1
        let revision = renderRevision
        isProcessing = true

        let currentParams = baseParams
        let currentOverlay = debugOverlay

        renderQueue.async {
            autoreleasepool {
                guard let inputImage = CIImage(image: sourceImage) else {
                    DispatchQueue.main.async {
                        if revision == renderRevision {
                            isProcessing = false
                        }
                    }
                    return
                }

                var correctedParams = currentParams.mappedForEngine()
                correctedParams = Self.applyPoseAndExpression(baseParams: correctedParams, metrics: tracking.metrics)

                let output = pipeline.process(
                    image: inputImage,
                    tracking: tracking,
                    params: correctedParams,
                    debugOverlay: currentOverlay
                )

                guard let cgImage = ciContext.createCGImage(output, from: output.extent) else {
                    DispatchQueue.main.async {
                        if revision == renderRevision {
                            isProcessing = false
                        }
                    }
                    return
                }

                let uiImage = UIImage(cgImage: cgImage, scale: 1.0, orientation: .up)
                DispatchQueue.main.async {
                    guard revision == renderRevision else { return }
                    renderedImage = uiImage
                    isProcessing = false
                }
            }
        }
    }

    private static func applyPoseAndExpression(baseParams: BeautyParams, metrics: FaceMetrics) -> BeautyParams {
        let yawFactor = 1 - min(abs(metrics.yawDegrees) / 35, 1)
        let pitchFactor = 1 - min(abs(metrics.pitchDegrees) / 30, 1)
        let poseFactor = max(0.2, yawFactor * pitchFactor)

        var corrected = baseParams
        corrected.faceSlim *= poseFactor
        corrected.jawSlim *= poseFactor
        corrected.eyeScale *= poseFactor
        corrected.noseSlim *= poseFactor

        if metrics.mouthOpen > 0.35 {
            corrected.jawSlim *= 0.5
        }
        if metrics.smile > 0.4 {
            corrected.faceSlim *= 0.75
        }
        if metrics.eyeOpen < 0.3 {
            corrected.eyeScale *= 0.2
        }

        return corrected
    }
}

private final class StillFaceAnalyzer {
    private let landmarker: FaceLandmarker?

    init() {
        landmarker = Self.makeLandmarker()
    }

    func analyze(image: UIImage) -> FaceTrackingState? {
        guard let landmarker,
              let mpImage = try? MPImage(uiImage: image),
              let result = try? landmarker.detect(image: mpImage),
              let face = result.faceLandmarks.first,
              !face.isEmpty else {
            return nil
        }

        let points = face.map { CGPoint(x: CGFloat($0.x), y: CGFloat($0.y)) }
        let metrics = computeFaceMetrics(landmarks: points)
        return FaceTrackingState(landmarks: points, metrics: metrics, smoothingAlpha: 0)
    }

    private static func makeLandmarker() -> FaceLandmarker? {
        guard let modelPath = Bundle.main.path(forResource: "face_landmarker", ofType: "task") else {
            return nil
        }

        let baseOptions = BaseOptions()
        baseOptions.modelAssetPath = modelPath

        let options = FaceLandmarkerOptions()
        options.baseOptions = baseOptions
        options.runningMode = .image
        options.numFaces = 1
        options.minFaceDetectionConfidence = 0.5
        options.minFacePresenceConfidence = 0.5
        options.minTrackingConfidence = 0.5

        return try? FaceLandmarker(options: options)
    }

    private func computeFaceMetrics(landmarks: [CGPoint]) -> FaceMetrics {
        func point(_ index: Int) -> CGPoint? {
            guard index >= 0, index < landmarks.count else { return nil }
            return landmarks[index]
        }

        func distance(_ a: Int, _ b: Int) -> CGFloat {
            guard let p1 = point(a), let p2 = point(b) else { return 0 }
            return hypot(p1.x - p2.x, p1.y - p2.y)
        }

        let mouthOpen = distance(13, 14) / max(distance(78, 308), 0.0001)
        let leftEyeOpen = distance(159, 145) / max(distance(33, 133), 0.0001)
        let rightEyeOpen = distance(386, 374) / max(distance(362, 263), 0.0001)
        let eyeOpen = ((leftEyeOpen + rightEyeOpen) * 0.5).clamped(to: 0...1)

        var smile: CGFloat = 0
        if let leftCorner = point(61),
           let rightCorner = point(291),
           let upperLip = point(13),
           let lowerLip = point(14),
           let forehead = point(10),
           let chin = point(152) {
            let lipCenterY = (upperLip.y + lowerLip.y) * 0.5
            let cornersY = (leftCorner.y + rightCorner.y) * 0.5
            let faceHeight = max(hypot(forehead.x - chin.x, forehead.y - chin.y), 0.0001)
            smile = ((lipCenterY - cornersY) / faceHeight * 6).clamped(to: 0...1)
        }

        var yawDegrees: CGFloat = 0
        if let left = point(234), let right = point(454), let nose = point(1) {
            let leftSpan = abs(nose.x - left.x)
            let rightSpan = abs(right.x - nose.x)
            let imbalance = abs(leftSpan - rightSpan) / max(leftSpan + rightSpan, 0.0001)
            yawDegrees = min(70, imbalance * 140)
        }

        var pitchDegrees: CGFloat = 0
        if let leftEyeTop = point(159),
           let rightEyeTop = point(386),
           let upperLip = point(13),
           let lowerLip = point(14),
           let nose = point(1) {
            let eyeCenterY = (leftEyeTop.y + rightEyeTop.y) * 0.5
            let mouthCenterY = (upperLip.y + lowerLip.y) * 0.5
            let ratio = (nose.y - eyeCenterY) / max(mouthCenterY - eyeCenterY, 0.0001)
            pitchDegrees = min(60, abs(ratio - 0.5) * 120)
        }

        var rollDegrees: CGFloat = 0
        if let leftEye = point(33), let rightEye = point(263) {
            rollDegrees = atan2(rightEye.y - leftEye.y, rightEye.x - leftEye.x) * 180.0 / .pi
        }

        return FaceMetrics(
            yawDegrees: yawDegrees,
            pitchDegrees: pitchDegrees,
            rollDegrees: rollDegrees,
            mouthOpen: mouthOpen,
            smile: smile,
            eyeOpen: eyeOpen
        )
    }
}

private struct ParameterSliderRow: View {
    let title: String
    @Binding var value: Double
    let range: ClosedRange<Double>
    var onEditingChanged: (Bool) -> Void = { _ in }

    var body: some View {
        HStack(spacing: 10) {
            Text(title)
                .font(.caption)
                .foregroundStyle(.white)
                .frame(width: 92, alignment: .leading)

            Slider(value: $value, in: range, onEditingChanged: onEditingChanged)
                .tint(.white)

            Text(String(format: "%.2f", value))
                .font(.caption.monospacedDigit())
                .foregroundStyle(.white.opacity(0.8))
                .frame(width: 38, alignment: .trailing)
        }
    }
}

private struct CameraView: UIViewRepresentable {
    var baseParams: BeautyParams
    var debugOverlay: Bool

    func makeUIView(context: Context) -> CameraMetalView {
        let view = CameraMetalView()
        view.updateBaseParams(baseParams)
        view.updateDebugOverlay(debugOverlay)
        return view
    }

    func updateUIView(_ uiView: CameraMetalView, context: Context) {
        uiView.updateBaseParams(baseParams)
        uiView.updateDebugOverlay(debugOverlay)
    }
}

final class CameraMetalView: UIView {
    private let captureSession = AVCaptureSession()
    private let sessionQueue = DispatchQueue(label: "BeautyCam.session.queue")
    private let outputQueue = DispatchQueue(label: "BeautyCam.video.output.queue")
    private let stateQueue = DispatchQueue(label: "BeautyCam.state.queue")
    private let imageQueue = DispatchQueue(label: "BeautyCam.image.queue")

    private let commandQueue: MTLCommandQueue
    private let ciContext: CIContext
    private let metalView: MTKView
    private let colorSpace = CGColorSpaceCreateDeviceRGB()

    private var baseParams: BeautyParams = .standard
    private var debugOverlay = false
    private var latestImage: CIImage?

    private var hasConfiguredSession = false
    private var hasRequestedAccess = false

    private let landmarkEngine = FaceLandmarkEngine()
    private let beautyPipeline = BeautyPipeline()

    override init(frame: CGRect) {
        guard let device = MTLCreateSystemDefaultDevice(),
              let queue = device.makeCommandQueue() else {
            fatalError("Metal device is required.")
        }

        commandQueue = queue
        ciContext = CIContext(mtlDevice: device)
        metalView = MTKView(frame: .zero, device: device)

        super.init(frame: frame)
        configureMetalView()
    }

    required init?(coder: NSCoder) {
        return nil
    }

    override func didMoveToWindow() {
        super.didMoveToWindow()
        if window != nil {
            startCameraIfNeeded()
        } else {
            stopSession()
        }
    }

    func updateBaseParams(_ params: BeautyParams) {
        stateQueue.sync {
            self.baseParams = params
        }
    }

    func updateDebugOverlay(_ enabled: Bool) {
        stateQueue.sync {
            self.debugOverlay = enabled
        }
    }

    private func configureMetalView() {
        metalView.translatesAutoresizingMaskIntoConstraints = false
        metalView.framebufferOnly = false
        metalView.isPaused = true
        metalView.enableSetNeedsDisplay = true
        metalView.contentMode = .scaleAspectFill
        metalView.delegate = self
        addSubview(metalView)

        NSLayoutConstraint.activate([
            metalView.topAnchor.constraint(equalTo: topAnchor),
            metalView.bottomAnchor.constraint(equalTo: bottomAnchor),
            metalView.leadingAnchor.constraint(equalTo: leadingAnchor),
            metalView.trailingAnchor.constraint(equalTo: trailingAnchor)
        ])
    }

    private func startCameraIfNeeded() {
        sessionQueue.async {
            guard Bundle.main.object(forInfoDictionaryKey: "NSCameraUsageDescription") != nil else {
                return
            }

            let status = AVCaptureDevice.authorizationStatus(for: .video)
            switch status {
            case .authorized:
                self.configureSessionIfNeeded()
                self.startSession()
            case .notDetermined:
                guard !self.hasRequestedAccess else { return }
                self.hasRequestedAccess = true
                AVCaptureDevice.requestAccess(for: .video) { granted in
                    if granted {
                        self.sessionQueue.async {
                            self.configureSessionIfNeeded()
                            self.startSession()
                        }
                    }
                }
            case .denied, .restricted:
                return
            @unknown default:
                return
            }
        }
    }

    private func configureSessionIfNeeded() {
        guard !hasConfiguredSession else { return }
        hasConfiguredSession = true

        captureSession.beginConfiguration()
        captureSession.sessionPreset = .hd1280x720

        guard let camera = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .front),
              let input = try? AVCaptureDeviceInput(device: camera),
              captureSession.canAddInput(input) else {
            captureSession.commitConfiguration()
            return
        }
        captureSession.addInput(input)

        let output = AVCaptureVideoDataOutput()
        output.videoSettings = [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA]
        output.alwaysDiscardsLateVideoFrames = true
        output.setSampleBufferDelegate(self, queue: outputQueue)

        guard captureSession.canAddOutput(output) else {
            captureSession.commitConfiguration()
            return
        }
        captureSession.addOutput(output)

        if let connection = output.connection(with: .video) {
            if connection.isVideoMirroringSupported {
                connection.isVideoMirrored = true
            }
            if #available(iOS 17.0, *) {
                if connection.isVideoRotationAngleSupported(90) {
                    connection.videoRotationAngle = 90
                }
            } else if connection.isVideoOrientationSupported {
                connection.videoOrientation = .portrait
            }
        }

        captureSession.commitConfiguration()
    }

    private func startSession() {
        if !captureSession.isRunning {
            captureSession.startRunning()
        }
    }

    private func stopSession() {
        sessionQueue.async {
            if self.captureSession.isRunning {
                self.captureSession.stopRunning()
            }
        }
    }

    private func aspectFillTransform(imageExtent: CGRect, drawableSize: CGSize) -> CGAffineTransform {
        guard imageExtent.width > 0, imageExtent.height > 0,
              drawableSize.width > 0, drawableSize.height > 0 else {
            return .identity
        }

        let scaleX = drawableSize.width / imageExtent.width
        let scaleY = drawableSize.height / imageExtent.height
        let scale = max(scaleX, scaleY)

        let scaledWidth = imageExtent.width * scale
        let scaledHeight = imageExtent.height * scale
        let x = (drawableSize.width - scaledWidth) * 0.5
        let y = (drawableSize.height - scaledHeight) * 0.5

        let translateToOrigin = CGAffineTransform(translationX: -imageExtent.origin.x, y: -imageExtent.origin.y)
        let scaleTransform = CGAffineTransform(scaleX: scale, y: scale)
        let centerTransform = CGAffineTransform(translationX: x, y: y)
        return translateToOrigin.concatenating(scaleTransform).concatenating(centerTransform)
    }
}

extension CameraMetalView: AVCaptureVideoDataOutputSampleBufferDelegate {
    func captureOutput(
        _ output: AVCaptureOutput,
        didOutput sampleBuffer: CMSampleBuffer,
        from connection: AVCaptureConnection
    ) {
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        let inputImage = CIImage(cvPixelBuffer: pixelBuffer)

        let (currentBaseParams, currentDebugOverlay) = stateQueue.sync {
            (baseParams.mappedForEngine(), debugOverlay)
        }
        let timestamp = CMSampleBufferGetPresentationTimeStamp(sampleBuffer)
        let timestampMs = Int((Double(timestamp.value) / Double(timestamp.timescale)) * 1000)

        let processed = landmarkEngine.process(
            pixelBuffer: pixelBuffer,
            timestampMs: timestampMs,
            baseParams: currentBaseParams
        )

        let outputImage = beautyPipeline.process(
            image: inputImage,
            tracking: processed.tracking,
            params: processed.params,
            debugOverlay: currentDebugOverlay
        )

        imageQueue.sync {
            self.latestImage = outputImage
        }

        DispatchQueue.main.async {
            self.metalView.draw()
        }
    }
}

extension CameraMetalView: MTKViewDelegate {
    func draw(in view: MTKView) {
        guard let drawable = view.currentDrawable else { return }
        guard let image = imageQueue.sync(execute: { latestImage }) else { return }
        guard let commandBuffer = commandQueue.makeCommandBuffer() else { return }

        let drawableSize = view.drawableSize
        var finalImage = image.transformed(by: aspectFillTransform(imageExtent: image.extent, drawableSize: drawableSize))
        finalImage = finalImage.cropped(to: CGRect(origin: .zero, size: drawableSize))

        ciContext.render(
            finalImage,
            to: drawable.texture,
            commandBuffer: commandBuffer,
            bounds: finalImage.extent,
            colorSpace: colorSpace
        )

        commandBuffer.present(drawable)
        commandBuffer.commit()
    }

    func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {
    }
}

#Preview {
    ContentView()
}
