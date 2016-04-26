#ifndef CUDAEDGEDETECTION_H
#define CUDAEDGEDETECTION_H

#include <QtWidgets/QMainWindow>
#include "ui_cudaedgedetection.h"

class CUDAEdgeDetection : public QMainWindow
{
	Q_OBJECT

public:
	CUDAEdgeDetection(QWidget *parent = 0);
	~CUDAEdgeDetection();

private slots:

void onSobelOperatorCPUButtonClicked();
void onSobelOperatorGPUButtonClicked();
void onGaussianBlurCPUButtonClicked();
void onGaussianBlurGPUButtonCLicked();

private:
	Ui::CUDAEdgeDetectionClass ui;
};

#endif // CUDAEDGEDETECTION_H
