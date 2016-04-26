#include "cuda.h"
#include "cuda_runtime.h"
#include "opencv2\core\core.hpp"
#include "utils.h"

const int BLOCKDIM = 16;

__global__
void sobelOperatorKernel(const uchar* d_input, uchar* d_output, const int numRows, const int numCols) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < 0 || x > numCols || y > numRows || y < 0)
        return;

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

    double magnitudeX = 0;
    double magnitudeY = 0;

    for (int j = 0; j < 3; ++j) {
        for (int i = 0; i < 3; ++i) {
            const int xFocus = i + x;
            const int yFocus = j + y;
            const int index = yFocus * numCols + xFocus;
            magnitudeX += d_input[index] * sobelX[i][j];
            magnitudeY += d_input[index] * sobelY[i][j];
        }
    }
    double magnitude = sqrt(magnitudeX * magnitudeX + magnitudeY * magnitudeY);

    if (magnitude < 0)
        magnitude = 0;
    if (magnitude > 255)
        magnitude = 255;

    d_output[y * numCols + x ] = magnitude;
}

void  sobelFilter(const uchar * h_input, uchar * h_output, const int numRows, const int numCols) 
{
    const int size = numRows * numCols * sizeof(uchar);

    uchar * d_input = nullptr;
    uchar * d_output = nullptr;

    checkCudaErrors(cudaMalloc(&d_input, size));
    checkCudaErrors(cudaMalloc(&d_output, size)); 
    checkCudaErrors(cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice));

    dim3 blockSize;
    blockSize.x = BLOCKDIM;
    blockSize.y = BLOCKDIM;

    dim3 gridSize;
    gridSize.x = numCols / BLOCKDIM + 1;
    gridSize.y = numRows / BLOCKDIM + 1;

    sobelOperatorKernel <<<gridSize, blockSize >>>(d_input, d_output, numRows, numCols);

    checkCudaErrors(cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost));

    checkCudaErrors(cudaFree(d_input));
    checkCudaErrors(cudaFree(d_output));
}