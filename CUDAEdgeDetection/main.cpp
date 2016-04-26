#include "cudaedgedetection.h"
#include <QtWidgets/QApplication>

int main(int argc, char *argv[])
{
	QApplication a(argc, argv);
	CUDAEdgeDetection w;
	w.show();
	return a.exec();
}
