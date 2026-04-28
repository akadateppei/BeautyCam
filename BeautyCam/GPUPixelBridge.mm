#import "GPUPixelBridge.h"
#include <gpupixel/gpupixel.h>
#include <vector>

static inline void ConvertBGRAtoRGBA(const uint8_t *src, int width, int height, int srcStride, std::vector<uint8_t> &dst) {
    dst.resize((size_t)width * (size_t)height * 4);
    uint8_t *out = dst.data();
    for (int y = 0; y < height; ++y) {
        const uint8_t *row = src + y * srcStride;
        for (int x = 0; x < width; ++x) {
            const uint8_t b = row[x * 4 + 0];
            const uint8_t g = row[x * 4 + 1];
            const uint8_t r = row[x * 4 + 2];
            const uint8_t a = row[x * 4 + 3];
            out[(y * width + x) * 4 + 0] = r;
            out[(y * width + x) * 4 + 1] = g;
            out[(y * width + x) * 4 + 2] = b;
            out[(y * width + x) * 4 + 3] = a;
        }
    }
}

@interface GPUPixelBridge () {
    std::shared_ptr<gpupixel::SourceRawData>     _source;
    std::shared_ptr<gpupixel::BeautyFaceFilter>  _beautyFilter;
    std::shared_ptr<gpupixel::FaceReshapeFilter> _reshapeFilter;
    std::shared_ptr<gpupixel::SinkRawData>       _sink;
    std::shared_ptr<gpupixel::FaceDetector>      _faceDetector;
    std::vector<float>                            _externalLandmarks;
    NSInteger                                     _frameCounter;
}
@property (nonatomic, assign) float eyeZoomLevel;
@property (nonatomic, assign) float faceSlimLevel;
@property (nonatomic, assign) float smoothLevel;
@property (nonatomic, assign) float whitenLevel;
@end

@implementation GPUPixelBridge

- (instancetype)initWithResourcePath:(NSString *)resourcePath {
    self = [super init];
    if (!self) return nil;

    NSString *modelsPath = [resourcePath stringByAppendingPathComponent:@"models/face_det.mars_model"];
    if (![[NSFileManager defaultManager] fileExistsAtPath:modelsPath]) {
        NSString *fallback = [NSBundle.mainBundle.bundlePath stringByAppendingPathComponent:@"Frameworks/gpupixel.framework"];
        NSString *fallbackModels = [fallback stringByAppendingPathComponent:@"models/face_det.mars_model"];
        if ([[NSFileManager defaultManager] fileExistsAtPath:fallbackModels]) {
            resourcePath = fallback;
        }
    }

    gpupixel::GPUPixel::SetResourcePath(resourcePath.UTF8String);
    NSLog(@"[GPUPixelBridge] resourcePath=%@", resourcePath);

    _source        = gpupixel::SourceRawData::Create();
    _beautyFilter  = gpupixel::BeautyFaceFilter::Create();
    _reshapeFilter = gpupixel::FaceReshapeFilter::Create();
    _sink          = gpupixel::SinkRawData::Create();
    _faceDetector  = gpupixel::FaceDetector::Create();

    if (_source && _beautyFilter && _reshapeFilter && _sink) {
        // GPUPixel official beauty chain: Source -> FaceReshape -> Beauty -> Sink
        _source->AddSink(_reshapeFilter);
        _reshapeFilter->AddSink(_beautyFilter);
        _beautyFilter->AddSink(_sink);
    }

    _smoothLevel = 0.7f;
    _whitenLevel = 0.3f;
    _frameCounter = 0;

    return self;
}

- (void)setEyeZoomLevel:(float)level  { _eyeZoomLevel  = level; }
- (void)setFaceSlimLevel:(float)level { _faceSlimLevel = level; }
- (void)setSmoothLevel:(float)level   { _smoothLevel = level; }
- (void)setWhitenLevel:(float)level   { _whitenLevel = level; }

- (void)setExternalFaceLandmarks:(nullable const float *)landmarks count:(NSInteger)count {
    if (!landmarks || count <= 0) {
        _externalLandmarks.clear();
    } else {
        _externalLandmarks.assign(landmarks, landmarks + (size_t)count);
    }
}

- (nullable CIImage *)processPixelBuffer:(CVPixelBufferRef)pixelBuffer {
    if (!_source || !_beautyFilter || !_reshapeFilter || !_sink || !_faceDetector) return nil;

    _beautyFilter->SetBlurAlpha(_smoothLevel);
    _beautyFilter->SetWhite(_whitenLevel);

    _reshapeFilter->SetEyeZoomLevel(_eyeZoomLevel);
    _reshapeFilter->SetFaceSlimLevel(_faceSlimLevel);

    CVPixelBufferLockBaseAddress(pixelBuffer, kCVPixelBufferLock_ReadOnly);

    const uint8_t *data = (const uint8_t *)CVPixelBufferGetBaseAddress(pixelBuffer);
    int width  = (int)CVPixelBufferGetWidth(pixelBuffer);
    int height = (int)CVPixelBufferGetHeight(pixelBuffer);
    int stride = (int)CVPixelBufferGetBytesPerRow(pixelBuffer);

    std::vector<uint8_t> rgbaData;
    ConvertBGRAtoRGBA(data, width, height, stride, rgbaData);

    std::vector<float> landmarks;
    if (!_externalLandmarks.empty()) {
        landmarks = _externalLandmarks;
    } else {
        landmarks = _faceDetector->Detect(
            rgbaData.data(), width, height, width * 4,
            gpupixel::GPUPIXEL_MODE_FMT_PICTURE,
            gpupixel::GPUPIXEL_FRAME_TYPE_RGBA
        );
    }
    _reshapeFilter->SetFaceLandmarks(landmarks);

    _frameCounter += 1;
    if (_frameCounter % 30 == 0) {
        NSLog(@"[GPUPixelBridge] landmarks=%lu eye=%.3f face=%.3f smooth=%.3f white=%.3f",
              (unsigned long)landmarks.size(),
              _eyeZoomLevel,
              _faceSlimLevel,
              _smoothLevel,
              _whitenLevel);
    }

    _source->ProcessData(rgbaData.data(), width, height, width * 4,
                         gpupixel::GPUPIXEL_FRAME_TYPE_RGBA);

    CVPixelBufferUnlockBaseAddress(pixelBuffer, kCVPixelBufferLock_ReadOnly);

    const uint8_t *rgba = _sink->GetRgbaBuffer();
    int outW = _sink->GetWidth();
    int outH = _sink->GetHeight();

    if (!rgba || outW == 0 || outH == 0) return nil;

    NSData *pixelData = [NSData dataWithBytes:rgba
                                       length:(NSUInteger)(outW * outH * 4)];
    CIImage *image = [CIImage imageWithBitmapData:pixelData
                                      bytesPerRow:(size_t)(outW * 4)
                                             size:CGSizeMake(outW, outH)
                                           format:kCIFormatRGBA8
                                       colorSpace:CGColorSpaceCreateDeviceRGB()];
    if (!image) return nil;

    CGAffineTransform flip = CGAffineTransformMake(1, 0, 0, -1, 0, (CGFloat)outH);
    return [image imageByApplyingTransform:flip];
}

@end
