#include "ImageManager.h"
#include <QtWidgets/QFileDialog>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>

ImageManager::ImageManager()
{

}

ImageManager::~ImageManager()
{

}

ImageManager* ImageManager::instance()
{
    static ImageManager instance;
    return &instance;
}

void ImageManager::PrepareForComputation(ComputationType computType)
{
    PrepareImage();
}

void ImageManager::PrepareImage()
{
   
}
