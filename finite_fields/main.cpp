#include <cstdio>
#include <algorithm> 
#include <execution>
#include <thrust/host_vector.h> // thrust::host_vector
#include <thrust/device_vector.h> // thrust::device_vector
#include <thrust/transform.h> // thrust::transform

#include <math.h>
#include <chrono>

#include "finite_field.hpp"
#include "utils.hpp"


int main(int argc, char** argv){
    // Set device
	int device_id = std::stoi(argv[1]);
    CUDA_SAFE_CALL(cudaSetDevice(device_id));

	// int samples;
	int num_degrees = std::stoi(argv[2]);
	int degree_step = 10;
	long size = 100e3;

	std::vector<double> cpu_times_us(num_degrees);
	std::vector<double> gpu_times_us(num_degrees);

	thrust::host_vector<finite_field<int>> vec(size);
	thrust::host_vector<finite_field<int>> vecOut(size);

	for (int i=0; i<size; i++) {
		vec[i] = finite_field<int>(13, i+10);
	}

	// Allocate device memory and copy host vectors to device
    thrust::device_vector<finite_field<int>> d_vec = vec;
    thrust::device_vector<finite_field<int>> d_vecOut = vecOut;

	for (int i=0; i<num_degrees; i++) {
		int degree = i*degree_step;

		// CPU
		auto startCPU = std::chrono::high_resolution_clock::now();
		std::transform(std::execution::par, vec.begin(), vec.end(), vecOut.begin(), BlackBox(degree));
		auto endCPU = std::chrono::high_resolution_clock::now();
		auto time_taken_chrono_CPU = std::chrono::duration_cast<std::chrono::microseconds>(endCPU - startCPU);
		double time_taken_CPU = time_taken_chrono_CPU.count();

		// GPU
		auto startGPU = std::chrono::high_resolution_clock::now();
		thrust::transform(d_vec.begin(), d_vec.end(), d_vecOut.begin(), BlackBox(degree));
		cudaDeviceSynchronize();
		auto endGPU = std::chrono::high_resolution_clock::now();
		auto time_taken_chrono_GPU = std::chrono::duration_cast<std::chrono::microseconds>(endGPU - startGPU);
		double time_taken_GPU = time_taken_chrono_GPU.count();

		// Checking
		thrust::host_vector<finite_field<int>> GPU_Result(d_vecOut);
		for(int j = 0; j < size; j++) {
		    if(vecOut[i].value() != GPU_Result[i].value()) {
		        printf("Error in result: c[%d] = %d (expected %d) \n", j, GPU_Result[j], vecOut[j]);
				return 1;
			}
		}

		cpu_times_us.at(i) = time_taken_CPU;
		gpu_times_us.at(i) = time_taken_GPU;

		printf("\33[2K\r"); // Clear current line
		std::cout << "Completed: " << i << std::flush; // Write progress
	}
	std::cout << std::endl;

	files::saveVec1D(cpu_times_us, "cpu_timings_us.csv");
	files::saveVec1D(gpu_times_us, "gpu_timings_us.csv");
	
	// printf("CPU: %f [s], GPU: %f [s]; GPU is %d times faster \n", time_taken_CPU*1e-6, time_taken_GPU*1e-6, (int)floor(time_taken_CPU/time_taken_GPU));

	return 0;
}
