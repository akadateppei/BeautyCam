import SwiftUI
import Metal

// Holds AR session and Metal renderer across SwiftUI view recompositions.
final class FaceBeautySession: ObservableObject {
    let sessionManager = ARFaceSessionManager()
    let renderer: MetalFaceRenderer?

    init() {
        guard let device = MTLCreateSystemDefaultDevice() else {
            renderer = nil
            return
        }
        renderer = try? MetalFaceRenderer(device: device, sessionManager: sessionManager)
    }

    func start() {
        guard ARFaceSessionManager.isSupported else { return }
        sessionManager.start()
    }

    func stop() {
        sessionManager.stop()
    }
}

struct ContentView: View {
    @StateObject private var session = FaceBeautySession()
    @State private var parameters = BeautyParameters.default
    @State private var showWireframe = false
    @State private var arNotSupported = false
    @GestureState private var isSuppressing = false

    private var suppressedParameters: BeautyParameters {
        isSuppressing ? BeautyParameters(
            skinSmooth: 0, faceSlim: 0, jawSharpness: 0, chinPoint: 0, eyeScale: 0,
            noseSlim: 0, noseWingSlim: 0, mouthAdjust: 0, overallStrength: 0
        ) : parameters
    }

    var body: some View {
        ZStack(alignment: .bottom) {
            if arNotSupported {
                unsupportedView
            } else if let renderer = session.renderer {
                MetalViewRepresentable(
                    renderer: renderer,
                    parameters: suppressedParameters,
                    showWireframe: showWireframe
                )
                .ignoresSafeArea()
                .gesture(
                    DragGesture(minimumDistance: 0)
                        .updating($isSuppressing) { _, state, _ in state = true }
                )
            } else {
                Color.black.ignoresSafeArea()
                Text("Metal initialization failed")
                    .foregroundStyle(.white)
            }

            if !arNotSupported {
                BeautyControlPanel(
                    parameters: $parameters,
                    showWireframe: $showWireframe
                )
                .padding(.bottom, 20)
            }
        }
        .background(.black)
        .onAppear {
            if ARFaceSessionManager.isSupported {
                session.start()
            } else {
                arNotSupported = true
            }
        }
        .onDisappear {
            session.stop()
        }
    }

    private var unsupportedView: some View {
        VStack(spacing: 16) {
            Image(systemName: "faceid")
                .font(.system(size: 60))
                .foregroundStyle(.white.opacity(0.7))
            Text("TrueDepth カメラが必要です")
                .font(.headline)
                .foregroundStyle(.white)
            Text("このアプリはTrueDepthフロントカメラを搭載したiPhoneが必要です。")
                .font(.subheadline)
                .foregroundStyle(.white.opacity(0.7))
                .multilineTextAlignment(.center)
                .padding(.horizontal)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .background(.black)
        .ignoresSafeArea()
    }
}
