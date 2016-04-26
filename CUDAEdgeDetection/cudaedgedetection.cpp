#include "cudaedgedetection.h"
#include "ComputeEngine.h"

CUDAEdgeDetection::CUDAEdgeDetection(QWidget *parent)
	: QMainWindow(parent)
{
	ui.setupUi(this);
    QObject::connect(ui.pushButton, SIGNAL(clicked()), this, SLOT(onSobelOperatorCPUButtonClicked()));
    QObject::connect(ui.pushButton_2, SIGNAL(clicked()), this, SLOT(onSobelOperatorGPUButtonClicked()));
    QObject::connect(ui.pushButton_3, SIGNAL(clicked()), this, SLOT(onGaussianBlurCPUButtonClicked()));
    QObject::connect(ui.pushButton_4, SIGNAL(clicked()), this, SLOT(onGaussianBlurGPUButtonCLicked()));

}

CUDAEdgeDetection::~CUDAEdgeDetection()
{

}

void CUDAEdgeDetection::onSobelOperatorCPUButtonClicked()
{
    ComputeEngine::instance()->Compute(EdgeDetecion, CPUCompute);
}

void CUDAEdgeDetection::onSobelOperatorGPUButtonClicked()
{
    ComputeEngine::instance()->Compute(EdgeDetecion, GPUCompute);
}

void CUDAEdgeDetection::onGaussianBlurCPUButtonClicked()
{
    ComputeEngine::instance()->Compute(Denoising, CPUCompute);
}

void CUDAEdgeDetection::onGaussianBlurGPUButtonCLicked()
{
    ComputeEngine::instance()->Compute(Denoising, GPUCompute);
}