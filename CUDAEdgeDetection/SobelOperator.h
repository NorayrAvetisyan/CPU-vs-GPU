#include "cuda_runtime.h"
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include "utils.h"

enum SobelCalculationType{SobelCPUCalculate, SobelGPUCalculate};

class SobelOperator{
public:
    SobelOperator();
    ~SobelOperator();
    void Calculate(cv::Mat& inputImage,  SobelCalculationType);
private:

    void CPUCalculation(const unsigned char* const input, unsigned char* const output,
                        const size_t numRows, const size_t numCols);

    void GPUCalculation(const uchar * const input, uchar* const output,
                        const size_t numRows, const size_t numCols);

    int GetArrayIndex(int x, int y, int offset);
};