import SwiftUI

struct BeautyControlPanel: View {
    @Binding var parameters: BeautyParameters
    @Binding var showWireframe: Bool

    var body: some View {
        VStack(spacing: 6) {
            Toggle("Wireframe", isOn: $showWireframe)
                .font(.caption)
                .tint(.green)
                .padding(.bottom, 2)

            SliderRow(title: "Smooth",     value: $parameters.skinSmooth)
            SliderRow(title: "Face Slim",  value: $parameters.faceSlim)
            SliderRow(title: "Jaw",        value: $parameters.jawSharpness)
            SliderRow(title: "Eye",        value: $parameters.eyeScale)
            SliderRow(title: "Nose",       value: $parameters.noseSlim)
            SliderRow(title: "Alar",       value: $parameters.noseWingSlim)
            SliderRow(title: "Mouth",      value: $parameters.mouthAdjust)
            SliderRow(title: "Symmetry",   value: $parameters.symmetry)
            SliderRow(title: "Overall",    value: $parameters.overallStrength)
        }
        .padding()
        .background(.ultraThinMaterial)
        .clipShape(RoundedRectangle(cornerRadius: 16, style: .continuous))
        .padding(.horizontal)
    }
}

private struct SliderRow: View {
    let title: String
    @Binding var value: Float

    var body: some View {
        HStack(spacing: 8) {
            Text(title)
                .font(.caption)
                .foregroundStyle(.white)
                .frame(width: 72, alignment: .leading)
            Slider(value: $value, in: 0...1)
                .tint(.white)
            Text(String(format: "%.2f", value))
                .font(.caption.monospacedDigit())
                .foregroundStyle(.white.opacity(0.8))
                .frame(width: 36, alignment: .trailing)
        }
    }
}
