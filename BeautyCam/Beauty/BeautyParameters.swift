import Foundation

struct BeautyParameters {
    var faceSlim: Float
    var jawSharpness: Float
    var eyeScale: Float
    var noseSlim: Float
    var noseWingSlim: Float
    var mouthAdjust: Float
    var symmetry: Float
    var overallStrength: Float

    static let `default` = BeautyParameters(
        faceSlim: 0.35,
        jawSharpness: 0.25,
        eyeScale: 0.30,
        noseSlim: 0.20,
        noseWingSlim: 0.0,
        mouthAdjust: 0.10,
        symmetry: 0.10,
        overallStrength: 1.0
    )
}
