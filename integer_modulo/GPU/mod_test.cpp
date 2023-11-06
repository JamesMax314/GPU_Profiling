#include "../../common/cuda_safe_call.hpp"

#include <stdio.h> // printf

// Classic modulo
__global__ void modulo(int n, int *a, int *b, int *c)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] % b[i];
}

// Witchcraft modulo
__global__ void witchcraft_modulo(int n, int *a, int *b, int *c)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] % b[i];
}

int main() 
{
    // Used for timing
    const int N_samples = 1e4;
    float time[N_samples]; // in ms
    cudaEvent_t start[N_samples];
    cudaEvent_t stop[N_samples];

    int device_id = 0;
    int n = 1e6; // number of elements
    int threads_per_block = 64;
    int blocks = (n+threads_per_block-1)/threads_per_block; // Make sure > 300 to avoid tail effects
    size_t vector_size = n*sizeof(int); // size of n int

    // Set device
    CUDA_SAFE_CALL(cudaSetDevice(device_id));

    // Setup profiling events
    for (int i=0; i<N_samples; i++) 
    {
        CUDA_SAFE_CALL( cudaEventCreate(&(start[i])) );
        CUDA_SAFE_CALL( cudaEventCreate(&(stop[i])) );
    }

    // Allocate host memory
    int *a, *b, *c;
    a = (int*) malloc(vector_size);
    b = (int*) malloc(vector_size);
    c = (int*) malloc(vector_size);
    
    // Allocate device memory
    int *d_a, *d_b, *d_c;
    CUDA_SAFE_CALL(cudaMalloc(&d_a, vector_size));
    CUDA_SAFE_CALL(cudaMalloc(&d_b, vector_size));
    CUDA_SAFE_CALL(cudaMalloc(&d_c, vector_size));
    
    // Initialise host vectors
    for (size_t i = 0; i < n; i++)
    {
        a[i] = 8*i+1;
        b[i] = 2*i+1;
    }

    // Copy host vectors to device
    CUDA_SAFE_CALL(cudaMemcpy(d_a, a, vector_size, cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(d_b, b, vector_size, cudaMemcpyHostToDevice));

    // Perform vector addition
    for (int i=0; i<N_samples; i++) 
    {
        CUDA_SAFE_CALL( cudaEventRecord(start[i], 0) );
        modulo<<< blocks, threads_per_block>>>(n, d_a, d_b, d_c);
        CUDA_SAFE_CALL( cudaEventRecord(stop[i], 0) );
    }
    
    // Block for GPU to catch up
    CUDA_SAFE_CALL(cudaPeekAtLastError());
    CUDA_SAFE_CALL(cudaDeviceSynchronize());

    // Get timing data back form GPU
    float total_time = 0;
    float total_time_sq = 0;
    for (int i=0; i<N_samples; i++) 
    {
        CUDA_SAFE_CALL( cudaEventElapsedTime(&(time[i]), start[i], stop[i]) );
        total_time += time[i]/n;
        total_time_sq += (time[i]/n)*(time[i]/n);
    }

    // Copy result back to host
    CUDA_SAFE_CALL(cudaMemcpy(c, d_c, vector_size, cudaMemcpyDeviceToHost));
    
    for(int i = 0; i < n; i++)
        if( c[i] != (int)(a[i] % b[i]) )
            printf("Error in result: c[%d] = %d (expected %d) \n", i, c[i], (int)(a[i] % b[i]));
    
    // Check a few results
    printf("c[0]    = %d\n", c[0]);
    printf("c[n-1]  = %d\n", c[n-1]);

    // Print timing results
    double average_time = total_time/double(N_samples);    
    double std_dev = std::sqrt((total_time_sq - N_samples*average_time*average_time)/(N_samples-1.0)); 
    printf("Average time to compute: %e s, Standard Deviation: %e s, %d samples \n", average_time*1e-3, std_dev*1e-3, N_samples);
    
    // Free memory
    CUDA_SAFE_CALL(cudaFree(d_a));
    CUDA_SAFE_CALL(cudaFree(d_b));
    CUDA_SAFE_CALL(cudaFree(d_c));
    free(a);
    free(b);
    free(c);

    return 0;
}
