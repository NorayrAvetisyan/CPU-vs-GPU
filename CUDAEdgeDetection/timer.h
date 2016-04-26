#ifndef GPU_TIMER_H__
#define GPU_TIMER_H__
#include <chrono>
using namespace std;
using  namespace chrono;
struct CPUTimer
{
  CPUTimer()
  {
      
  }

  ~CPUTimer()
  {
    
  }

  void Start()
  {
      start = high_resolution_clock::now();
  }

  void Stop()
  {
      end = high_resolution_clock::now();
  }

  float Elapsed()
  {
      duration<double> time_span = duration_cast<duration<double>>(end - start);
      return time_span.count();
  }
private:
    high_resolution_clock::time_point start;
    high_resolution_clock::time_point end;
};

struct GPUTimer
{
    cudaEvent_t start;
    cudaEvent_t stop;

    GPUTimer()
    {
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
    }

    ~GPUTimer()
    {
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    void Start()
    {
        cudaEventRecord(start, 0);
    }

    void Stop()
    {
        cudaEventRecord(stop, 0);
    }

    float Elapsed()
    {
        float elapsed;
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsed, start, stop);
        return elapsed / 1000;
    }
};

#endif  /* GPU_TIMER_H__ */
