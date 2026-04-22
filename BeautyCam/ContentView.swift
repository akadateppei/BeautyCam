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

    var body: some View {
        ZStack(alignment: .bottom) {
            CameraView(isFaceBeautyEnabled: $isFaceBeautyEnabled)
                .ignoresSafeArea()

            Toggle(isOn: $isFaceBeautyEnabled) {
                Text("美顔（目拡大）")
                    .font(.headline)
                    .foregroundStyle(.white)
            }
            .toggleStyle(.switch)
            .padding(.horizontal, 20)
            .padding(.vertical, 14)
            .background(.black.opacity(0.45), in: Capsule())
            .padding(.bottom, 28)
        }
        .background(.black)
    }
}

private struct CameraView: UIViewRepresentable {
    @Binding var isFaceBeautyEnabled: Bool

    func makeUIView(context: Context) -> CameraMetalView {
        let view = CameraMetalView()
        view.updateFaceBeautyEnabled(isFaceBeautyEnabled)
        return view
    }

    func updateUIView(_ uiView: CameraMetalView, context: Context) {
        uiView.updateFaceBeautyEnabled(isFaceBeautyEnabled)
    }
}

final class CameraMetalView: UIView {
    private struct EyeWarp {
        let center: CGPoint
        let radius: CGFloat
    }

    private let captureSession = AVCaptureSession()
    private let sessionQueue = DispatchQueue(label: "BeautyCam.session.queue")
    private let outputQueue = DispatchQueue(label: "BeautyCam.video.output.queue")

    private let stateQueue = DispatchQueue(label: "BeautyCam.state.queue")
    private var _isFaceBeautyEnabled = true
    private var latestImage: CIImage?
    private var eyeWarps: [EyeWarp] = []
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
        let state = stateQueue.sync { (_isFaceBeautyEnabled, eyeWarps) }
        let isFaceBeautyEnabled = state.0
        let currentEyeWarps = state.1
        guard isFaceBeautyEnabled, !currentEyeWarps.isEmpty else { return image }

        var output = image
        for eyeWarp in currentEyeWarps {
            let filter = CIFilter.bumpDistortion()
            filter.inputImage = output
            filter.center = eyeWarp.center
            filter.radius = Float(eyeWarp.radius)
            filter.scale = 0.25
            output = filter.outputImage ?? output
        }
        return output
    }

    private func updateEyeWarpsIfNeeded(pixelBuffer: CVPixelBuffer, imageSize: CGSize) {
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
                self.eyeWarps = []
            }
            return
        }

        let newWarps = makeEyeWarps(from: face, imageSize: imageSize)
        stateQueue.async {
            self.eyeWarps = newWarps
        }
    }

    private func makeEyeWarps(from face: VNFaceObservation, imageSize: CGSize) -> [EyeWarp] {
        guard let landmarks = face.landmarks else { return [] }

        var warps: [EyeWarp] = []
        if let leftEye = landmarks.leftEye, let leftWarp = makeEyeWarp(from: leftEye, faceBoundingBox: face.boundingBox, imageSize: imageSize) {
            warps.append(leftWarp)
        }
        if let rightEye = landmarks.rightEye, let rightWarp = makeEyeWarp(from: rightEye, faceBoundingBox: face.boundingBox, imageSize: imageSize) {
            warps.append(rightWarp)
        }
        return warps
    }

    private func makeEyeWarp(
        from region: VNFaceLandmarkRegion2D,
        faceBoundingBox: CGRect,
        imageSize: CGSize
    ) -> EyeWarp? {
        guard region.pointCount > 0 else { return nil }

        let faceRect = CGRect(
            x: faceBoundingBox.origin.x * imageSize.width,
            y: faceBoundingBox.origin.y * imageSize.height,
            width: faceBoundingBox.size.width * imageSize.width,
            height: faceBoundingBox.size.height * imageSize.height
        )

        let points: [CGPoint] = (0..<region.pointCount).map { index in
            let p = region.normalizedPoints[index]
            return CGPoint(
                x: faceRect.origin.x + CGFloat(p.x) * faceRect.width,
                y: faceRect.origin.y + CGFloat(p.y) * faceRect.height
            )
        }

        var sumX: CGFloat = 0
        var sumY: CGFloat = 0
        for point in points {
            sumX += point.x
            sumY += point.y
        }
        let center = CGPoint(x: sumX / CGFloat(points.count), y: sumY / CGFloat(points.count))

        var maxDistance: CGFloat = 0
        for point in points {
            let dx = point.x - center.x
            let dy = point.y - center.y
            let distance = sqrt(dx * dx + dy * dy)
            maxDistance = max(maxDistance, distance)
        }

        let radius = max(20, maxDistance * 2.4)
        return EyeWarp(center: center, radius: radius)
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
        updateEyeWarpsIfNeeded(pixelBuffer: pixelBuffer, imageSize: baseImage.extent.size)
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
