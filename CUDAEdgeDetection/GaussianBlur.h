#include "cuda_runtime.h"
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include "utils.h"

enum GaussianCalculationType{GaussianCPUCalculate, GaussianGPUCalculate};

class GaussianBLur{
public:
    GaussianBLur(int filterWidth = 3, float kernelSigma = 2.0);
    ~GaussianBLur();
    void Calculate(cv::Mat& inputImage, GaussianCalculationType calculationType);
private:
    int filterWidth_;
    float* filter_;
    float kernelSigma_;
    void BuildFilter();
    void ChannelConvolution(const unsigned char* const channel,
                            unsigned char* const channelBlurred,
                            const size_t numRows, const size_t numCols);

    void CPUCalculation(const uchar4* const rgbaImage, uchar4 *const outputImage,
                        const size_t numRows, const size_t numCols);

    void GPUCalculation(const uchar4 * const inputImageRGBA,
                        uchar4* const outputImageRGBA, const size_t numRows, 
                        const size_t numCols);
};