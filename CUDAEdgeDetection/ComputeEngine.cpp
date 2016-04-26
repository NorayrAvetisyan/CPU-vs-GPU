#include "ComputeEngine.h"
#include <QtWidgets/QFileDialog>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>

ComputeEngine::ComputeEngine()
{

}

ComputeEngine::~ComputeEngine()
{

}

ComputeEngine* ComputeEngine::instance()
{
    static ComputeEngine instance;
    return &instance;
}

void ComputeEngine::Compute(OperationType operationType, ComputeType computeType)
{
    QString fileName = QFileDialog::getOpenFileName(nullptr, "Select image to open.", QDir::homePath(), "Images (*.png *.xpm *.jpg)");
    if (!fileName.isEmpty())
    {
        cv::Mat image;
        switch (operationType)
        {
        case Denoising:
        {
            image = cv::imread(fileName.toStdString(), CV_LOAD_IMAGE_COLOR);
            cv::Mat rgbaImage;
            cv::cvtColor(image, rgbaImage, CV_BGR2RGBA);
            if (image.empty()) {
                std::cerr << "Couldn't open file: " << fileName.toStdString() << std::endl;
                exit(1);
            }
            switch (computeType)
            {
            case CPUCompute:
                gaussianBlur_.Calculate(rgbaImage, GaussianCPUCalculate);
                break;
            case GPUCompute:
                gaussianBlur_.Calculate(rgbaImage, GaussianGPUCalculate);
                break;
            default:
                break;
            }
            break;
        }
        case EdgeDetecion:
        {
            image = cv::imread(fileName.toStdString(), CV_LOAD_IMAGE_GRAYSCALE);
            if (image.empty()) {
                std::cerr << "Couldn't open file: " << fileName.toStdString() << std::endl;
                exit(1);
            }
            switch (computeType)
            {
            case CPUCompute:
                sobelOperator_.Calculate(image, SobelCPUCalculate);
                break;
            case GPUCompute:
                sobelOperator_.Calculate(image, SobelGPUCalculate);
                break;
            default:
                break;
            }
            break;
        }
        default:
            break;
        }
    }
}