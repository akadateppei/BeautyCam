//
//  ContentView.swift
//  BeautyCam
//
//  Created by AKADA TEPPEI on 2026/04/22.
//

import AVFoundation
import CoreImage
import CoreImage.CIFilterBuiltins
import MetalKit
import SwiftUI
import UIKit
import Vision

struct ContentView: View {
    @State private var isFaceBeautyEnabled = true
    @State private var eyeSize: Double = 0.25
    @State private var noseSize: Double = 0.20
    @State private var faceLine: Double = 0.20

    var body: some View {
        ZStack(alignment: .bottom) {
            CameraView(
                isFaceBeautyEnabled: $isFaceBeautyEnabled,
                eyeSize: $eyeSize,
                noseSize: $noseSize,
                faceLine: $faceLine
            )
                .ignoresSafeArea()

            VStack(spacing: 10) {
                Toggle(isOn: $isFaceBeautyEnabled) {
                    Text("美顔ON")
                        .font(.headline)
                        .foregroundStyle(.white)
                }
                .toggleStyle(.switch)

                BeautySliderRow(title: "目の大きさ", value: $eyeSize, range: 0...0.45)
                BeautySliderRow(title: "小鼻サイズ", value: $noseSize, range: 0...0.45)
                BeautySliderRow(title: "フェイスライン", value: $faceLine, range: 0...0.45)
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

private struct BeautySliderRow: View {
    let title: String
    @Binding var value: Double
    let range: ClosedRange<Double>

    var body: some View {
        HStack(spacing: 10) {
            Text(title)
                .font(.subheadline)
                .foregroundStyle(.white)
                .frame(width: 100, alignment: .leading)

            Slider(value: $value, in: range)
                .tint(.white)
        }
    }
}

private struct CameraView: UIViewRepresentable {
    @Binding var isFaceBeautyEnabled: Bool
    @Binding var eyeSize: Double
    @Binding var noseSize: Double
    @Binding var faceLine: Double

    func makeUIView(context: Context) -> CameraMetalView {
        let view = CameraMetalView()
        view.updateFaceBeautyEnabled(isFaceBeautyEnabled)
        view.updateAdjustments(eyeSize: CGFloat(eyeSize), noseSize: CGFloat(noseSize), faceLine: CGFloat(faceLine))
        return view
    }

    func updateUIView(_ uiView: CameraMetalView, context: Context) {
        uiView.updateFaceBeautyEnabled(isFaceBeautyEnabled)
        uiView.updateAdjustments(eyeSize: CGFloat(eyeSize), noseSize: CGFloat(noseSize), faceLine: CGFloat(faceLine))
    }
}

final class CameraMetalView: UIView {
    private struct FeatureAdjustments {
        var eyeSize: CGFloat = 0.25
        var noseSize: CGFloat = 0.20
        var faceLine: CGFloat = 0.20
    }

    private struct EyeWarp {
        let center: CGPoint
        let radius: CGFloat
    }

    private struct FaceWarpState {
        var eyes: [EyeWarp] = []
        var nose: EyeWarp?
        var chin: EyeWarp?
    }

    private let captureSession = AVCaptureSession()
    private let sessionQueue = DispatchQueue(label: "BeautyCam.session.queue")
    private let outputQueue = DispatchQueue(label: "BeautyCam.video.output.queue")

    private let stateQueue = DispatchQueue(label: "BeautyCam.state.queue")
    private var _isFaceBeautyEnabled = true
    private var latestImage: CIImage?
    private var faceWarpState = FaceWarpState()
    private var featureAdjustments = FeatureAdjustments()
    private var frameCounter = 0

    private let metalDevice: MTLDevice
    private let commandQueue: MTLCommandQueue
    private let ciContext: CIContext
    private let metalView: MTKView
    private let colorSpace = CGColorSpaceCreateDeviceRGB()
    private let visionSequenceHandler = VNSequenceRequestHandler()

    private var hasConfiguredSession = false
    private var hasRequestedAccess = false

    override init(frame: CGRect) {
        guard let device = MTLCreateSystemDefaultDevice(),
              let queue = device.makeCommandQueue() else {
            fatalError("Metal device is required.")
        }

        metalDevice = device
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

    func updateFaceBeautyEnabled(_ enabled: Bool) {
        stateQueue.async {
            self._isFaceBeautyEnabled = enabled
        }
    }

    func updateAdjustments(eyeSize: CGFloat, noseSize: CGFloat, faceLine: CGFloat) {
        stateQueue.async {
            self.featureAdjustments = FeatureAdjustments(
                eyeSize: eyeSize,
                noseSize: noseSize,
                faceLine: faceLine
            )
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
                // Avoid immediate termination when the privacy description key is missing.
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
        output.videoSettings = [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_420YpCbCr8BiPlanarFullRange]
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

    private func processFrame(_ image: CIImage) -> CIImage {
        let state = stateQueue.sync { (_isFaceBeautyEnabled, faceWarpState, featureAdjustments) }
        let isFaceBeautyEnabled = state.0
        let warpState = state.1
        let adjustments = state.2
        guard isFaceBeautyEnabled else { return image }

        var output = image
        for eyeWarp in warpState.eyes {
            let filter = CIFilter.bumpDistortion()
            filter.inputImage = output
            filter.center = eyeWarp.center
            filter.radius = Float(eyeWarp.radius)
            filter.scale = Float(adjustments.eyeSize)
            output = filter.outputImage ?? output
        }

        if let noseWarp = warpState.nose, adjustments.noseSize > 0 {
            let filter = CIFilter.pinchDistortion()
            filter.inputImage = output
            filter.center = noseWarp.center
            filter.radius = Float(noseWarp.radius)
            filter.scale = Float(adjustments.noseSize * 0.9)
            output = filter.outputImage ?? output
        }

        if let chinWarp = warpState.chin, adjustments.faceLine > 0 {
            let lowerPinch = CIFilter.pinchDistortion()
            lowerPinch.inputImage = output
            lowerPinch.center = chinWarp.center
            lowerPinch.radius = Float(chinWarp.radius)
            lowerPinch.scale = Float(adjustments.faceLine * 0.95)
            output = lowerPinch.outputImage ?? output

            let upperPinch = CIFilter.pinchDistortion()
            upperPinch.inputImage = output
            upperPinch.center = CGPoint(x: chinWarp.center.x, y: chinWarp.center.y + chinWarp.radius * 0.55)
            upperPinch.radius = Float(chinWarp.radius * 0.75)
            upperPinch.scale = Float(adjustments.faceLine * 0.45)
            output = upperPinch.outputImage ?? output
        }

        return output
    }

    private func updateFaceWarpsIfNeeded(pixelBuffer: CVPixelBuffer, imageSize: CGSize) {
        let shouldRunDetection = stateQueue.sync {
            frameCounter += 1
            return frameCounter % 4 == 0
        }
        guard shouldRunDetection else { return }

        let request = VNDetectFaceLandmarksRequest()

        do {
            try visionSequenceHandler.perform([request], on: pixelBuffer, orientation: .up)
        } catch {
            return
        }

        guard let face = request.results?.first else {
            stateQueue.async {
                self.faceWarpState = FaceWarpState()
            }
            return
        }

        let newWarps = makeFaceWarpState(from: face, imageSize: imageSize)
        stateQueue.async {
            self.faceWarpState = newWarps
        }
    }

    private func makeFaceWarpState(from face: VNFaceObservation, imageSize: CGSize) -> FaceWarpState {
        guard let landmarks = face.landmarks else { return FaceWarpState() }
        let faceRect = CGRect(
            x: face.boundingBox.origin.x * imageSize.width,
            y: face.boundingBox.origin.y * imageSize.height,
            width: face.boundingBox.size.width * imageSize.width,
            height: face.boundingBox.size.height * imageSize.height
        )

        var state = FaceWarpState()
        if let leftEye = landmarks.leftEye, let leftWarp = makeFeatureWarp(from: leftEye, faceRect: faceRect, radiusMultiplier: 2.4) {
            state.eyes.append(leftWarp)
        }
        if let rightEye = landmarks.rightEye, let rightWarp = makeFeatureWarp(from: rightEye, faceRect: faceRect, radiusMultiplier: 2.4) {
            state.eyes.append(rightWarp)
        }

        if let nose = landmarks.nose, let noseWarp = makeNoseWarp(from: nose, faceRect: faceRect) {
            state.nose = noseWarp
        } else {
            state.nose = EyeWarp(
                center: CGPoint(x: faceRect.midX, y: faceRect.minY + faceRect.height * 0.50),
                radius: max(16, faceRect.width * 0.09)
            )
        }

        state.chin = makeChinWarp(landmarks: landmarks, faceRect: faceRect)
        return state
    }

    private func makeNoseWarp(from region: VNFaceLandmarkRegion2D, faceRect: CGRect) -> EyeWarp? {
        let points = makePoints(from: region, faceRect: faceRect)
        guard !points.isEmpty else { return nil }

        let minY = points.map(\.y).min() ?? 0
        let maxY = points.map(\.y).max() ?? 0
        let cutoffY = minY + (maxY - minY) * 0.45
        let lowerNosePoints = points.filter { $0.y <= cutoffY }
        let target = lowerNosePoints.isEmpty ? points : lowerNosePoints

        let center = centroid(of: target)
        let baseRadius = maxDistance(from: center, points: target)
        let radius = max(14, min(faceRect.width * 0.13, baseRadius * 1.8))
        return EyeWarp(center: center, radius: radius)
    }

    private func makeChinWarp(landmarks: VNFaceLandmarks2D, faceRect: CGRect) -> EyeWarp {
        if let contour = landmarks.faceContour {
            let points = makePoints(from: contour, faceRect: faceRect)
            if let chinPoint = points.min(by: { $0.y < $1.y }) {
                return EyeWarp(
                    center: CGPoint(x: chinPoint.x, y: chinPoint.y + faceRect.height * 0.03),
                    radius: max(30, faceRect.width * 0.20)
                )
            }
        }

        return EyeWarp(
            center: CGPoint(x: faceRect.midX, y: faceRect.minY + faceRect.height * 0.08),
            radius: max(30, faceRect.width * 0.20)
        )
    }

    private func makeFeatureWarp(
        from region: VNFaceLandmarkRegion2D,
        faceRect: CGRect,
        radiusMultiplier: CGFloat
    ) -> EyeWarp? {
        let points = makePoints(from: region, faceRect: faceRect)
        guard !points.isEmpty else { return nil }

        let center = centroid(of: points)
        let maxDistance = maxDistance(from: center, points: points)

        let radius = max(20, maxDistance * radiusMultiplier)
        return EyeWarp(center: center, radius: radius)
    }

    private func makePoints(from region: VNFaceLandmarkRegion2D, faceRect: CGRect) -> [CGPoint] {
        guard region.pointCount > 0 else { return [] }
        return (0..<region.pointCount).map { index in
            let p = region.normalizedPoints[index]
            return CGPoint(
                x: faceRect.origin.x + CGFloat(p.x) * faceRect.width,
                y: faceRect.origin.y + CGFloat(p.y) * faceRect.height
            )
        }
    }

    private func centroid(of points: [CGPoint]) -> CGPoint {
        let sum = points.reduce(CGPoint.zero) { partial, point in
            CGPoint(x: partial.x + point.x, y: partial.y + point.y)
        }
        return CGPoint(x: sum.x / CGFloat(points.count), y: sum.y / CGFloat(points.count))
    }

    private func maxDistance(from center: CGPoint, points: [CGPoint]) -> CGFloat {
        points.reduce(CGFloat.zero) { currentMax, point in
            let dx = point.x - center.x
            let dy = point.y - center.y
            return max(currentMax, sqrt(dx * dx + dy * dy))
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
        let baseImage = CIImage(cvPixelBuffer: pixelBuffer)
        updateFaceWarpsIfNeeded(pixelBuffer: pixelBuffer, imageSize: baseImage.extent.size)
        let filtered = processFrame(baseImage)

        stateQueue.async {
            self.latestImage = filtered
        }

        DispatchQueue.main.async {
            self.metalView.draw()
        }
    }
}

extension CameraMetalView: MTKViewDelegate {
    func draw(in view: MTKView) {
        guard let drawable = view.currentDrawable else { return }
        let image: CIImage? = stateQueue.sync { latestImage }
        guard let image else { return }
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
