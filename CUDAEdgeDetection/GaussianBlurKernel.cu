#include <device_functions.hpp>
#include "utils.h"

const int BLOCKDIM = 16;

__global__ 
void gaussianBlur(const unsigned char* const inputChannel,
                  unsigned char* const outputChannel,
                  int numRows, int numCols,
                  const float* const filter, const int filterWidth)
{

    const int2 thread2DPos = make_int2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);
    if (thread2DPos.x >= numCols || thread2DPos.y >= numRows)
        return;
    const int thread1DPos = thread2DPos.y * numCols + thread2DPos.x;

    const int row = thread2DPos.y;
    const int col = thread2DPos.x;

    float result = 0.0;
    for (int filter_i = -filterWidth / 2; filter_i <= filterWidth / 2; ++filter_i) {
        for (int filter_j = -filterWidth / 2; filter_j <= filterWidth / 2; ++filter_j) {
            int image_i = min(max(row + filter_i, 0), (numRows - 1));
            int image_j = min(max(col + filter_j, 0), (numCols - 1));

            float imageValue = (float)inputChannel[image_i * numCols + image_j];
            float filterValue = filter[(filter_i + filterWidth / 2) * filterWidth + filter_j + filterWidth / 2];

            result += imageValue * filterValue;
        }
    }

    outputChannel[thread1DPos] = (unsigned char)result;
}

__global__
void separateChannels(const uchar4* const inputImageRGBA,
                      int numRows, int numCols,
                      unsigned char* const redChannel,
                      unsigned char* const greenChannel,
                      unsigned char* const blueChannel)
{

    const int2 thread2DPos = make_int2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);
    if (thread2DPos.x >= numCols || thread2DPos.y >= numRows)
        return;
    const int thread1DPos = thread2DPos.y * numCols + thread2DPos.x;
    redChannel[thread1DPos] = inputImageRGBA[thread1DPos].x;
    greenChannel[thread1DPos] = inputImageRGBA[thread1DPos].y;
    blueChannel[thread1DPos] = inputImageRGBA[thread1DPos].z;
}

__global__
void recombineChannels(const unsigned char* const redChannel,
                       const unsigned char* const greenChannel,
                       const unsigned char* const blueChannel,
                       uchar4* const outputImageRGBA,
                       int numRows, int numCols)
{
    const int2 thread2DPos = make_int2(blockIdx.x * blockDim.x + threadIdx.x,
        blockIdx.y * blockDim.y + threadIdx.y);

    const int thread1DPos = thread2DPos.y * numCols + thread2DPos.x;

    if (thread2DPos.x >= numCols || thread2DPos.y >= numRows)
        return;

    unsigned char red = redChannel[thread1DPos];
    unsigned char green = greenChannel[thread1DPos];
    unsigned char blue = blueChannel[thread1DPos];

    uchar4 outputPixel = make_uchar4(red, green, blue, 255);

    outputImageRGBA[thread1DPos] = outputPixel;
}

unsigned char *d_red, *d_green, *d_blue;
float         *d_filter;

void allocateMemoryAndCopyToGPU(const size_t numRowsImage, const size_t numColsImage,
                                const float* const h_filter, const size_t filterWidth)
{

    checkCudaErrors(cudaMalloc(&d_red, sizeof(unsigned char) * numRowsImage * numColsImage));
    checkCudaErrors(cudaMalloc(&d_green, sizeof(unsigned char) * numRowsImage * numColsImage));
    checkCudaErrors(cudaMalloc(&d_blue, sizeof(unsigned char) * numRowsImage * numColsImage));

    checkCudaErrors(cudaMalloc(&d_filter, sizeof(float) * filterWidth * filterWidth));

    checkCudaErrors(cudaMemcpy(d_filter, h_filter, sizeof(float) * filterWidth * filterWidth, cudaMemcpyHostToDevice));

}

void gaussianBlur(const uchar4 * const h_inputImageRGBA, uchar4 * const d_inputImageRGBA,
                  uchar4* const d_outputImageRGBA, const size_t numRows, const size_t numCols,
                  unsigned char *d_redBlurred, unsigned char *d_greenBlurred, unsigned char *d_blueBlurred,
                  const int filterWidth, const float* filter)
{
    allocateMemoryAndCopyToGPU(numRows, numCols, filter, filterWidth);
    dim3 blockSize;
    blockSize.x = BLOCKDIM;
    blockSize.y = BLOCKDIM;

    dim3 gridSize;
    gridSize.x = numCols / BLOCKDIM + 1;
    gridSize.y = numRows / BLOCKDIM + 1;

    separateChannels <<<gridSize, blockSize >>>(d_inputImageRGBA, numRows, numCols, d_red, d_green, d_blue);
    cudaDeviceSynchronize(); 
    checkCudaErrors(cudaGetLastError());

    gaussianBlur <<<gridSize, blockSize >>>(d_red, d_redBlurred, numRows, numCols, d_filter, filterWidth);
    cudaDeviceSynchronize(); 
    checkCudaErrors(cudaGetLastError());

    gaussianBlur <<<gridSize, blockSize >>>(d_green, d_greenBlurred, numRows, numCols, d_filter, filterWidth);
    cudaDeviceSynchronize();
    checkCudaErrors(cudaGetLastError());

    gaussianBlur <<<gridSize, blockSize >>>(d_blue, d_blueBlurred, numRows, numCols, d_filter, filterWidth);
    cudaDeviceSynchronize();
    checkCudaErrors(cudaGetLastError());

    recombineChannels <<<gridSize, blockSize >>>(d_redBlurred, d_greenBlurred, d_blueBlurred,
                                                   d_outputImageRGBA, numRows, numCols);
    cudaDeviceSynchronize(); 
    checkCudaErrors(cudaGetLastError());

}

void cleanup() {
    checkCudaErrors(cudaFree(d_red));
    checkCudaErrors(cudaFree(d_green));
    checkCudaErrors(cudaFree(d_blue));
    checkCudaErrors(cudaFree(d_filter));
}
