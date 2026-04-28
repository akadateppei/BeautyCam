import SwiftUI
import MetalKit

struct MetalViewRepresentable: UIViewRepresentable {
    let renderer: MetalFaceRenderer
    var parameters: BeautyParameters
    var showWireframe: Bool

    func makeUIView(context: Context) -> MTKView {
        let view = MTKView(frame: .zero, device: MTLCreateSystemDefaultDevice())
        view.delegate              = renderer
        view.colorPixelFormat      = .bgra8Unorm
        view.framebufferOnly       = false
        view.isPaused              = false
        view.preferredFramesPerSecond = 60
        view.backgroundColor       = .black
        return view
    }

    func updateUIView(_ uiView: MTKView, context: Context) {
        renderer.parameters   = parameters
        renderer.showWireframe = showWireframe
    }
}
