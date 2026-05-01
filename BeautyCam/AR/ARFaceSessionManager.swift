import ARKit

final class ARFaceSessionManager: NSObject, ARFrameProviding {
    let session = ARSession()

    private(set) var latestFrame: ARFrame?
    private(set) var latestFaceAnchor: ARFaceAnchor?

    // Stored to allow clean resume after interruption without full reset
    private var activeConfiguration: ARFaceTrackingConfiguration?

    static var isSupported: Bool {
        ARFaceTrackingConfiguration.isSupported
    }

    func start() {
        guard ARFaceTrackingConfiguration.isSupported else {
            print("[ARFaceSessionManager] Face tracking not supported on this device.")
            return
        }
        let config = ARFaceTrackingConfiguration()
        config.isLightEstimationEnabled = false  // unused; disable to save GPU
        activeConfiguration = config
        session.delegate = self
        session.run(config, options: [.resetTracking, .removeExistingAnchors])
    }

    func stop() {
        session.pause()
    }
}

extension ARFaceSessionManager: ARSessionDelegate {

    // Capture every frame for rendering
    func session(_ session: ARSession, didUpdate frame: ARFrame) {
        latestFrame = frame
    }

    // Face anchor appears
    func session(_ session: ARSession, didAdd anchors: [ARAnchor]) {
        if let face = anchors.compactMap({ $0 as? ARFaceAnchor }).first {
            latestFaceAnchor = face
        }
    }

    // Face anchor updates each frame
    func session(_ session: ARSession, didUpdate anchors: [ARAnchor]) {
        if let face = anchors.compactMap({ $0 as? ARFaceAnchor }).first {
            latestFaceAnchor = face
        }
    }

    // Face anchor lost
    func session(_ session: ARSession, didRemove anchors: [ARAnchor]) {
        if anchors.contains(where: { $0 is ARFaceAnchor }) {
            latestFaceAnchor = nil
        }
    }

    func session(_ session: ARSession, didFailWithError error: Error) {
        print("[ARFaceSessionManager] Session error: \(error.localizedDescription)")
    }

    func sessionWasInterrupted(_ session: ARSession) {
        print("[ARFaceSessionManager] Session interrupted.")
    }

    // Resume without resetting tracking — preserves face anchor continuity
    func sessionInterruptionEnded(_ session: ARSession) {
        guard let config = activeConfiguration else { return }
        session.run(config)
    }
}
