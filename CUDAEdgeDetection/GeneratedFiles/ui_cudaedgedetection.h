/********************************************************************************
** Form generated from reading UI file 'cudaedgedetection.ui'
**
** Created by: Qt User Interface Compiler version 5.6.0
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_CUDAEDGEDETECTION_H
#define UI_CUDAEDGEDETECTION_H

#include <QtCore/QVariant>
#include <QtWidgets/QAction>
#include <QtWidgets/QApplication>
#include <QtWidgets/QButtonGroup>
#include <QtWidgets/QHeaderView>
#include <QtWidgets/QLabel>
#include <QtWidgets/QMainWindow>
#include <QtWidgets/QMenuBar>
#include <QtWidgets/QPushButton>
#include <QtWidgets/QStatusBar>
#include <QtWidgets/QToolBar>
#include <QtWidgets/QWidget>

QT_BEGIN_NAMESPACE

class Ui_CUDAEdgeDetectionClass
{
public:
    QWidget *centralWidget;
    QPushButton *pushButton;
    QPushButton *pushButton_2;
    QLabel *label;
    QPushButton *pushButton_3;
    QPushButton *pushButton_4;
    QLabel *label_2;
    QMenuBar *menuBar;
    QToolBar *mainToolBar;
    QStatusBar *statusBar;

    void setupUi(QMainWindow *CUDAEdgeDetectionClass)
    {
        if (CUDAEdgeDetectionClass->objectName().isEmpty())
            CUDAEdgeDetectionClass->setObjectName(QStringLiteral("CUDAEdgeDetectionClass"));
        CUDAEdgeDetectionClass->resize(408, 288);
        centralWidget = new QWidget(CUDAEdgeDetectionClass);
        centralWidget->setObjectName(QStringLiteral("centralWidget"));
        pushButton = new QPushButton(centralWidget);
        pushButton->setObjectName(QStringLiteral("pushButton"));
        pushButton->setGeometry(QRect(75, 160, 120, 40));
        pushButton_2 = new QPushButton(centralWidget);
        pushButton_2->setObjectName(QStringLiteral("pushButton_2"));
        pushButton_2->setGeometry(QRect(200, 160, 120, 40));
        label = new QLabel(centralWidget);
        label->setObjectName(QStringLiteral("label"));
        label->setGeometry(QRect(120, 10, 181, 31));
        QFont font;
        font.setPointSize(15);
        font.setBold(true);
        font.setWeight(75);
        label->setFont(font);
        pushButton_3 = new QPushButton(centralWidget);
        pushButton_3->setObjectName(QStringLiteral("pushButton_3"));
        pushButton_3->setGeometry(QRect(75, 50, 120, 40));
        pushButton_4 = new QPushButton(centralWidget);
        pushButton_4->setObjectName(QStringLiteral("pushButton_4"));
        pushButton_4->setGeometry(QRect(200, 50, 120, 40));
        label_2 = new QLabel(centralWidget);
        label_2->setObjectName(QStringLiteral("label_2"));
        label_2->setGeometry(QRect(120, 120, 171, 31));
        label_2->setFont(font);
        CUDAEdgeDetectionClass->setCentralWidget(centralWidget);
        menuBar = new QMenuBar(CUDAEdgeDetectionClass);
        menuBar->setObjectName(QStringLiteral("menuBar"));
        menuBar->setGeometry(QRect(0, 0, 408, 21));
        CUDAEdgeDetectionClass->setMenuBar(menuBar);
        mainToolBar = new QToolBar(CUDAEdgeDetectionClass);
        mainToolBar->setObjectName(QStringLiteral("mainToolBar"));
        CUDAEdgeDetectionClass->addToolBar(Qt::TopToolBarArea, mainToolBar);
        statusBar = new QStatusBar(CUDAEdgeDetectionClass);
        statusBar->setObjectName(QStringLiteral("statusBar"));
        CUDAEdgeDetectionClass->setStatusBar(statusBar);

        retranslateUi(CUDAEdgeDetectionClass);

        QMetaObject::connectSlotsByName(CUDAEdgeDetectionClass);
    } // setupUi

    void retranslateUi(QMainWindow *CUDAEdgeDetectionClass)
    {
        CUDAEdgeDetectionClass->setWindowTitle(QApplication::translate("CUDAEdgeDetectionClass", "Edge Detection (CPU vs GPU)", 0));
        pushButton->setText(QApplication::translate("CUDAEdgeDetectionClass", "Sobel Operator (CPU)", 0));
        pushButton_2->setText(QApplication::translate("CUDAEdgeDetectionClass", "Sobel Operator (GPU)", 0));
        label->setText(QApplication::translate("CUDAEdgeDetectionClass", "Image Denoising", 0));
        pushButton_3->setText(QApplication::translate("CUDAEdgeDetectionClass", "Gaussian Blur (CPU)", 0));
        pushButton_4->setText(QApplication::translate("CUDAEdgeDetectionClass", "Gaussian Blur (GPU)", 0));
        label_2->setText(QApplication::translate("CUDAEdgeDetectionClass", "Edge Detection", 0));
    } // retranslateUi

};

namespace Ui {
    class CUDAEdgeDetectionClass: public Ui_CUDAEdgeDetectionClass {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_CUDAEDGEDETECTION_H
