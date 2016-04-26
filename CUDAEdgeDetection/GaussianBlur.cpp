#include "GaussianBlur.h"
#include <cmath>
#include <assert.h>
#include <algorithm>
#include "utils.h"
#include "cuda.h"
#include "timer.h"
#include "QtWidgets/QMessageBox"

void gaussianBlur(const uchar4 * const h_inputImageRGBA, uchar4 * const d_inputImageRGBA,
    uchar4* const d_outputImageRGBA, const size_t numRows, const size_t numCols,
    unsigned char *d_redBlurred, unsigned char *d_greenBlurred, unsigned char *d_blueBlurred,
    const int filterWidth, const float* filter);

GaussianBLur::GaussianBLur(int filterWidth, float kernelSigma)
{
    filterWidth_ =  filterWidth;
    assert(filterWidth_ % 2 == 1);
    kernelSigma_ = kernelSigma;
    filter_ = new float[filterWidth_ * filterWidth_];
    assert(filter_ != nullptr);
    BuildFilter();
}

GaussianBLur::~GaussianBLur()
{
    delete filter_;
}

void GaussianBLur::BuildFilter()
{
    float filterSum = 0.0;

    for (int i = -filterWidth_ / 2; i <= filterWidth_ / 2; ++i) {
        for (int j = -filterWidth_ / 2; j <= filterWidth_ / 2; ++j) {
            float filterValue = expf(-(float)(j * j + i * i) / (2.0 * kernelSigma_ * kernelSigma_));
            (filter_)[(i + filterWidth_ / 2) * filterWidth_ + j + filterWidth_ / 2] = filterValue;
            filterSum += filterValue;
        }
    }

    float normalizationFactor = 1.0 / filterSum;

    for (int i = -filterWidth_ / 2; i <= filterWidth_ / 2; ++i) {
        for (int j = -filterWidth_ / 2; j <= filterWidth_ / 2; ++j) {
            (filter_)[(i + filterWidth_ / 2) * filterWidth_ + j + filterWidth_ / 2] *= normalizationFactor;
        }
    }
}

void GaussianBLur::ChannelConvolution(const unsigned char* const channel,
                                      unsigned char* const channelBlurred,
                                      const size_t numRows, const size_t numCols)
{
    
    for (int i = 0; i < (int)numRows; ++i) {
        for (int j = 0; j < (int)numCols; ++j) {
            float result = 0.0;
            for (int filter_i = -filterWidth_ / 2; filter_i <= filterWidth_ / 2; ++filter_i) {
                for (int filter_j = -filterWidth_ / 2; filter_j <= filterWidth_ / 2; ++filter_j) {

                    int image_i = std::min(std::max(i + filter_i, 0), static_cast<int>(numRows - 1));
                    int image_j = std::min(std::max(j + filter_j, 0), static_cast<int>(numCols - 1));

                    float imageValue = static_cast<float>(channel[image_i * numCols + image_j]);
                    float filterValue = filter_[(filter_i + filterWidth_ / 2) * filterWidth_ + filter_j + filterWidth_ / 2];

                    result += imageValue * filterValue;
                }
            }

            channelBlurred[i * numCols + j] = result;
        }
    }
}

void GaussianBLur::CPUCalculation(const uchar4* const rgbaImage, uchar4 *const outputImage,
                                  const size_t numRows, const size_t numCols)
{
    unsigned char *red = new unsigned char[numRows * numCols];
    unsigned char *blue = new unsigned char[numRows * numCols];
    unsigned char *green = new unsigned char[numRows * numCols];

    unsigned char *redBlurred = new unsigned char[numRows * numCols];
    unsigned char *blueBlurred = new unsigned char[numRows * numCols];
    unsigned char *greenBlurred = new unsigned char[numRows * numCols];

    for (size_t i = 0; i < numRows * numCols; ++i) {
        uchar4 rgba = rgbaImage[i];
        red[i] = rgba.x;
        green[i] = rgba.y;
        blue[i] = rgba.z;
    }

    ChannelConvolution(red, redBlurred, numRows, numCols);
    ChannelConvolution(green, greenBlurred, numRows, numCols);
    ChannelConvolution(blue, blueBlurred, numRows, numCols);

    for (size_t i = 0; i < numRows * numCols; ++i) {
        uchar4 rgba = make_uchar4(redBlurred[i], greenBlurred[i], blueBlurred[i], 255);
        outputImage[i] = rgba;
    }

    delete[] red;
    delete[] green;
    delete[] blue;

    delete[] redBlurred;
    delete[] greenBlurred;
    delete[] blueBlurred;
}

void GaussianBLur::GPUCalculation(const uchar4 * const inputImageRGBA,
                                 uchar4* const outputImageRGBA, 
                                 const size_t numRows, const size_t numCols)
{
    const size_t numPixels = numRows * numCols;
    uchar4 *d_inputImageRGBA;
    uchar4 *d_outputImageRGBA;
    checkCudaErrors(cudaMalloc(&d_inputImageRGBA, sizeof(uchar4) * numPixels));
    checkCudaErrors(cudaMalloc(&d_outputImageRGBA, sizeof(uchar4) * numPixels));
    checkCudaErrors(cudaMemset(d_outputImageRGBA, 0, numPixels * sizeof(uchar4))); 

    //copy input array to the GPU
    checkCudaErrors(cudaMemcpy(d_inputImageRGBA, inputImageRGBA, sizeof(uchar4) * numPixels, cudaMemcpyHostToDevice));
    unsigned char *d_redBlurred, *d_greenBlurred, *d_blueBlurred;
    checkCudaErrors(cudaMalloc(&d_redBlurred, sizeof(unsigned char) * numPixels));
    checkCudaErrors(cudaMalloc(&d_greenBlurred, sizeof(unsigned char) * numPixels));
    checkCudaErrors(cudaMalloc(&d_blueBlurred, sizeof(unsigned char) * numPixels));
    checkCudaErrors(cudaMemset(d_redBlurred, 0, sizeof(unsigned char) * numPixels));
    checkCudaErrors(cudaMemset(d_greenBlurred, 0, sizeof(unsigned char) * numPixels));
    checkCudaErrors(cudaMemset(d_blueBlurred, 0, sizeof(unsigned char) * numPixels));
    gaussianBlur(inputImageRGBA, d_inputImageRGBA, d_outputImageRGBA, numRows, numCols, d_redBlurred, d_greenBlurred, d_blueBlurred, filterWidth_, filter_);
    checkCudaErrors(cudaMemcpy(outputImageRGBA, d_outputImageRGBA, sizeof(uchar4) * numPixels, cudaMemcpyDeviceToHost));
}

void GaussianBLur::Calculate(cv::Mat& inputImageRGBA, GaussianCalculationType calculationType)
{
    cv::Mat imageOutputRGBA;
    imageOutputRGBA.create(inputImageRGBA.rows, inputImageRGBA.cols, CV_8UC4);

    if (!inputImageRGBA.isContinuous() || !imageOutputRGBA.isContinuous()) {
        std::cerr << "Images aren't continuous!! Exiting." << std::endl;
        exit(1);
    }

    GPUTimer gpuTimer;
    CPUTimer cpuTimer;
    float elepsedTime;
    uchar4* h_inputImageRGBA = (uchar4 *)inputImageRGBA.ptr<unsigned char>(0);
    uchar4* h_outputImageRGBA = (uchar4 *)imageOutputRGBA.ptr<unsigned char>(0);
    switch (calculationType)
    {
    case GaussianCPUCalculate:
    {
        cpuTimer.Start();
        CPUCalculation(h_inputImageRGBA, h_outputImageRGBA, inputImageRGBA.rows, inputImageRGBA.cols);
        cpuTimer.Stop();
        elepsedTime = cpuTimer.Elapsed();
    }
        break;
    case GaussianGPUCalculate:
    {
        gpuTimer.Start();
        GPUCalculation(h_inputImageRGBA, h_outputImageRGBA, inputImageRGBA.rows, inputImageRGBA.cols);
        gpuTimer.Stop();
        elepsedTime = gpuTimer.Elapsed();
    }
        break;
    default:
        break;
    }

    cv::Mat output(inputImageRGBA.rows, inputImageRGBA.cols,  CV_8UC4, (void*)h_outputImageRGBA);
    cv::Mat imageOutputBGR;
    cv::cvtColor(output, imageOutputBGR, CV_RGBA2BGR);

    cv::namedWindow("GaussianBlur", cv::WINDOW_NORMAL);
    cv::imshow("GaussianBlur", imageOutputBGR);
    cv::moveWindow("GaussianBlur", 0, 0);
    cv::resizeWindow("GaussianBlur", 640, 480);
    QMessageBox::information(nullptr, "Operation time", "Processing time - " + QString::number(elepsedTime) + "  second", QMessageBox::Ok);
    cv::waitKey(0);
    cv::destroyAllWindows();
}