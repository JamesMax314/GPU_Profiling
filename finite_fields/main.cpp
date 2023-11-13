#include <cstdio>
#include <algorithm> 
#include <thrust/host_vector.h> // thrust::host_vector
#include <thrust/device_vector.h> // thrust::device_vector
#include <thrust/transform.h> // thrust::transform

#include <chrono>

#include "finite_field.hpp"

finite_field<int> ff(100, 2);
finite_field<int> ff2(13, 12);
// black_box bb;
// black_box bbGPU;

int main(){
    // Set device
	int device_id = 1;
    CUDA_SAFE_CALL(cudaSetDevice(device_id));

	long size = 1e6;
	thrust::host_vector<finite_field<int>> vec(size);
	thrust::host_vector<finite_field<int>> vecOut(size);

	for (int i=0; i<size; i++) {
		vec[i] = finite_field<int>(13, i+10);
	}

	// Allocate device memory and copy host vectors to device
    thrust::device_vector<finite_field<int>> d_vec = vec;
    thrust::device_vector<finite_field<int>> d_vecOut = vecOut;

	// CPU
	auto startCPU = std::chrono::high_resolution_clock::now();
	thrust::transform(vec.begin(), vec.end(), vecOut.begin(), black_box());
	auto endCPU = std::chrono::high_resolution_clock::now();
	auto time_taken_chrono_CPU = std::chrono::duration_cast<std::chrono::microseconds>(endCPU - startCPU);
	double time_taken_CPU = time_taken_chrono_CPU.count();

	// GPU
	auto startGPU = std::chrono::high_resolution_clock::now();
	thrust::transform(d_vec.begin(), d_vec.end(), d_vecOut.begin(), black_box());
	cudaDeviceSynchronize();
	auto endGPU = std::chrono::high_resolution_clock::now();
	auto time_taken_chrono_GPU = std::chrono::duration_cast<std::chrono::microseconds>(endGPU - startGPU);
	double time_taken_GPU = time_taken_chrono_GPU.count();

	// Checking
	thrust::host_vector<finite_field<int>> GPU_Result(d_vecOut);
	for(int i = 0; i < size; i++)
        if(vecOut[i].value() != GPU_Result[i].value()) {
            printf("Error in result: c[%d] = %d (expected %d) \n", i, GPU_Result[i], vecOut[i]);
		}
	
	printf("CPU: %f [s], GPU: %f [s]; GPU is %d times faster \n", time_taken_CPU*1e-6, time_taken_GPU*1e-6, (int)floor(time_taken_CPU/time_taken_GPU));

	return 0;
}
