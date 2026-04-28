import ARKit

final class ARFaceSessionManager: NSObject, ARFrameProviding {
    let session = ARSession()

    private(set) var latestFrame: ARFrame?
    private(set) var latestFaceAnchor: ARFaceAnchor?

    var onFaceLost: (() -> Void)?

    static var isSupported: Bool {
        ARFaceTrackingConfiguration.isSupported
    }

    func start() {
        guard ARFaceTrackingConfiguration.isSupported else {
            print("[ARFaceSessionManager] Face tracking is not supported on this device.")
            return
        }
        let configuration = ARFaceTrackingConfiguration()
        configuration.isLightEstimationEnabled = true
        session.delegate = self
        session.run(configuration, options: [.resetTracking, .removeExistingAnchors])
    }

    func stop() {
        session.pause()
    }
}

extension ARFaceSessionManager: ARSessionDelegate {
    func session(_ session: ARSession, didUpdate frame: ARFrame) {
        latestFrame = frame
        let anchor = frame.anchors.compactMap { $0 as? ARFaceAnchor }.first
        if anchor == nil, latestFaceAnchor != nil {
            onFaceLost?()
        }
        latestFaceAnchor = anchor
    }

    func session(_ session: ARSession, didFailWithError error: Error) {
        print("[ARFaceSessionManager] Session error: \(error.localizedDescription)")
    }

    func sessionWasInterrupted(_ session: ARSession) {
        print("[ARFaceSessionManager] Session interrupted.")
    }

    func sessionInterruptionEnded(_ session: ARSession) {
        start()
    }
}
