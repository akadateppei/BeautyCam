#import <CoreImage/CoreImage.h>
#import <CoreVideo/CoreVideo.h>
#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

@interface GPUPixelBridge : NSObject

- (instancetype)initWithResourcePath:(NSString *)resourcePath NS_DESIGNATED_INITIALIZER;
- (instancetype)init NS_UNAVAILABLE;

- (void)setEyeZoomLevel:(float)level;
- (void)setFaceSlimLevel:(float)level;
- (void)setSmoothLevel:(float)level;
- (void)setWhitenLevel:(float)level;

/// Pass pre-computed face landmarks (x/y normalized 0-1, interleaved: x0,y0,x1,y1,...).
/// count = total float count. Pass count=0 to fall back to internal face detector.
- (void)setExternalFaceLandmarks:(nullable const float *)landmarks count:(NSInteger)count;

- (nullable CIImage *)processPixelBuffer:(CVPixelBufferRef)pixelBuffer;

@end

NS_ASSUME_NONNULL_END
