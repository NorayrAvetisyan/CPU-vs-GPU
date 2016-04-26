#include "SobelOperator.h"
#include <cmath>
#include <assert.h>
#include <algorithm>
#include "cuda_runtime.h"
#include "cuda.h"
#include "timer.h"
#include <QtWidgets/QMessageBox>

using namespace cv;

void sobelFilter(const uchar * h_input, uchar * h_output, const int numRows, const int numCols);

SobelOperator::SobelOperator()
{
}

SobelOperator::~SobelOperator()
{
}

int SobelOperator::GetArrayIndex(int x, int y, int offset)
{
    return y * offset + x;
}
void SobelOperator::CPUCalculation(const unsigned char* input, unsigned char * output,
                                   const size_t numRows, const size_t numCols)
{
    const int sobelX[3][3] = {
            { -1, 0, 1 },
            { -2, 0, 2 },
            { -1, 0, 1 }
    };
    
    const int sobelY[3][3] = {
            { -1, -2, -1 },
            { 0, 0, 0 },
            { 1, 2, 1 }
    };

    int height = numRows;
    int width = numCols;
    for (int y = 1; y < numRows - 2; ++y) {
        for (int x = 1; x < numCols - 2; ++x) {
            const double pixelX = (double)(
                (sobelX[0][0] * input[GetArrayIndex(x - 1, y - 1, numCols)]) +
                (sobelX[0][1] * input[GetArrayIndex(x, y - 1, numCols)]) +
                (sobelX[0][2] * input[GetArrayIndex(x + 1, y - 1, numCols)]) +
                (sobelX[1][0] * input[GetArrayIndex(x - 1, y, numCols)]) +
                (sobelX[1][1] * input[GetArrayIndex(x, y, numCols)]) +
                (sobelX[1][2] * input[GetArrayIndex(x + 1, y, numCols)]) +
                (sobelX[2][0] * input[GetArrayIndex(x - 1, y + 1, numCols)]) +
                (sobelX[2][1] * input[GetArrayIndex(x, y + 1, numCols)]) +
                (sobelX[2][2] * input[GetArrayIndex(x + 1, y + 1, numCols)])
                );
            const double pixelY = (double)(
                (sobelY[0][0] * input[GetArrayIndex(x - 1, y - 1, numCols)]) +
                (sobelY[0][1] * input[GetArrayIndex(x, y - 1, numCols)]) +
                (sobelY[0][2] * input[GetArrayIndex(x + 1, y - 1, numCols)]) +
                (sobelY[1][0] * input[GetArrayIndex(x - 1, y, numCols)]) +
                (sobelY[1][1] * input[GetArrayIndex(x, y, numCols)]) +
                (sobelY[1][2] * input[GetArrayIndex(x + 1, y, numCols)]) +
                (sobelY[2][0] * input[GetArrayIndex(x - 1, y + 1, numCols)]) +
                (sobelY[2][1] * input[GetArrayIndex(x, y + 1, numCols)]) +
                (sobelY[2][2] * input[GetArrayIndex(x + 1, y + 1, numCols)])
                );

            double magnitude = sqrt(pixelX * pixelX + pixelY * pixelY);

            if (magnitude < 0)
                magnitude = 0;
            if (magnitude > 255)
                magnitude = 255;

            output[y * numCols + x] = magnitude;
        }
    }
}

void SobelOperator::GPUCalculation(const uchar* const input, uchar* const output,
                                   const size_t numRows, const size_t numCols)
{
    sobelFilter(input, output, numRows, numCols);
}

void SobelOperator::Calculate(cv::Mat& inputImageGrayScale, SobelCalculationType calculationType)
{
    cv::Mat outputImage = inputImageGrayScale.clone();

    GPUTimer gpuTimer;
    CPUTimer cpuTimer;
    float elepsedTime;
    unsigned char* h_inputImage = inputImageGrayScale.data;
    unsigned char* h_outputImage = outputImage.data; 
    switch (calculationType)
    {
    case SobelCPUCalculate:
    {
        cpuTimer.Start();
        CPUCalculation(h_inputImage, h_outputImage, inputImageGrayScale.rows, inputImageGrayScale.cols);
        cpuTimer.Stop();
        elepsedTime = cpuTimer.Elapsed();
    }
        break;
    case SobelGPUCalculate:
    {
        gpuTimer.Start();
        GPUCalculation(h_inputImage, h_outputImage, inputImageGrayScale.rows, inputImageGrayScale.cols);
        gpuTimer.Stop();
        elepsedTime = gpuTimer.Elapsed();
    }
        break;
    default:
        break;
    }
    cv::namedWindow("SobelOperator", cv::WINDOW_NORMAL);
    cv::imshow("SobelOperator", outputImage);
    cv::moveWindow("SobelOperator", 0, 0);
    cv::resizeWindow("SobelOperator", 640, 480);
    QMessageBox::information(nullptr, "Operation time", "Processing time - " + QString::number(elepsedTime) + "  second", QMessageBox::Ok);
    cv::waitKey(0);
    cv::destroyAllWindows();
}

