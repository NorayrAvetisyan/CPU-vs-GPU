#include "GaussianBlur.h"
#include "SobelOperator.h"
enum ComputeType{CPUCompute, GPUCompute};
enum OperationType{Denoising, EdgeDetecion};
class ComputeEngine{
public:
    static ComputeEngine* instance();
    ~ComputeEngine();
    void Compute(OperationType operationType, ComputeType computeType);
private:
    ComputeEngine();
    ComputeEngine(const ComputeEngine& obj);
    ComputeEngine& operator=(const ComputeEngine& obj);
    GaussianBLur gaussianBlur_;
    SobelOperator sobelOperator_;
};