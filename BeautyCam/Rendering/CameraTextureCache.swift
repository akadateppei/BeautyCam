import Metal
import CoreVideo

final class CameraTextureCache {
    private var cache: CVMetalTextureCache?

    init?(device: MTLDevice) {
        var c: CVMetalTextureCache?
        let status = CVMetalTextureCacheCreate(nil, nil, device, nil, &c)
        guard status == kCVReturnSuccess, let cache = c else { return nil }
        self.cache = cache
    }

    func textures(from pixelBuffer: CVPixelBuffer) -> (y: MTLTexture?, cbcr: MTLTexture?) {
        guard let cache else { return (nil, nil) }
        let width  = CVPixelBufferGetWidth(pixelBuffer)
        let height = CVPixelBufferGetHeight(pixelBuffer)

        var yRef: CVMetalTexture?
        CVMetalTextureCacheCreateTextureFromImage(
            nil, cache, pixelBuffer, nil,
            .r8Unorm, width, height, 0, &yRef)

        var cbcrRef: CVMetalTexture?
        CVMetalTextureCacheCreateTextureFromImage(
            nil, cache, pixelBuffer, nil,
            .rg8Unorm, width / 2, height / 2, 1, &cbcrRef)

        let y    = yRef.flatMap { CVMetalTextureGetTexture($0) }
        let cbcr = cbcrRef.flatMap { CVMetalTextureGetTexture($0) }
        return (y, cbcr)
    }

    func flush() {
        guard let cache else { return }
        CVMetalTextureCacheFlush(cache, 0)
    }
}
