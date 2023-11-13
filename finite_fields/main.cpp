#include <cstdio>
#include <algorithm> 
#include <thrust/host_vector.h> // thrust::host_vector
#include <thrust/device_vector.h> // thrust::device_vector
#include <thrust/transform.h> // thrust::transform

#include "finite_field.hpp"

finite_field<int> ff(100, 2);
finite_field<int> ff2(13, 12);
// black_box bb;
// black_box bbGPU;

int main(){
	int device_id = 1;
    
    // Set device
    CUDA_SAFE_CALL(cudaSetDevice(device_id));

	int size = 10000000;
	thrust::host_vector<finite_field<int>> vec(size);
	thrust::host_vector<finite_field<int>> vecOut(size);
	thrust::host_vector<finite_field<int>> GPU_Result(size);
	for (int i=0; i<size; i++) {
		vec[i] = finite_field<int>(13, i+10);
	}

	// Allocate device memory and copy host vectors to device
    thrust::device_vector<finite_field<int>> d_vec = vec;
    thrust::device_vector<finite_field<int>> d_vecOut = vecOut;

	thrust::transform(vec.begin(), vec.end(), vecOut.begin(), black_box());
	thrust::transform(d_vec.begin(), d_vec.end(), d_vecOut.begin(), black_box());
	
	GPU_Result = d_vecOut;

	for (int i=0; i<size; i++) {
		printf("CPU: %d GPU: %d \n", vecOut[i], GPU_Result[i]);
	}

	return 0;
}
