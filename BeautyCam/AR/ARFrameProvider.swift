import ARKit

protocol ARFrameProviding: AnyObject {
    var latestFrame: ARFrame? { get }
    var latestFaceAnchor: ARFaceAnchor? { get }
}
