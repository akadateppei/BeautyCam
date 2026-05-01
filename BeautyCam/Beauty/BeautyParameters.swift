import Foundation

struct BeautyParameters {
    var skinSmooth: Float
    var faceSlim: Float
    var jawSharpness: Float
    var eyeScale: Float
    var noseSlim: Float
    var noseWingSlim: Float
    var mouthAdjust: Float
    var symmetry: Float
    var overallStrength: Float

    static let `default` = BeautyParameters(
        skinSmooth: 0.50,   // literature: #1 most impactful beauty parameter
        faceSlim: 0.30,
        jawSharpness: 0.20, // literature: "slightly more defined"
        eyeScale: 0.35,
        noseSlim: 0.20,
        noseWingSlim: 0.0,
        mouthAdjust: 0.10,
        symmetry: 0.10,
        overallStrength: 1.0
    )
}
