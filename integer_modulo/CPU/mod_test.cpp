#include <stdio.h>
#include <ctime>
#include <array>
#include <cmath>
#include <numeric>
#include <chrono>

int main(){
    const int N_samples = 10000;
    const int N_ops = 100000;

    double total_time = 0.0;
    double total_time_sq = 0.0;

    int sum = 0;
 
    std::array<int, N_ops> a, b;
    for (size_t i = 0; i < N_ops; i++)
    {
        a.at(i) = 8*i+1;
        b.at(i) = 2*i+1;
    }

    for(int i =0; i<N_samples; i++){
	std::array<int, N_ops> c;
        auto previous_time = std::chrono::high_resolution_clock::now();
        for(int j=0; j<N_ops; j++){
	    c.at(j) = a.at(j) % b.at(j);
        }
        auto current_time = std::chrono::high_resolution_clock::now();
        auto time_taken_chrono = std::chrono::duration_cast<std::chrono::microseconds>(current_time - previous_time);
        double time_taken = time_taken_chrono.count()/double(N_ops);
        total_time += time_taken;
        total_time_sq += time_taken*time_taken;
        
        sum += std::accumulate(c.begin(), c.end(), 0);

    }    

    double average_time = total_time/double(N_samples);    
    double std_dev = std::sqrt((total_time_sq - N_samples*average_time*average_time)/(N_samples-1.0)); 

    printf("Average time to compute: %e ns, Standard Deviation: %e ns, %d samples \n", average_time*1e+3, std_dev*1e+3, N_samples);
    printf("%d \n", sum);

    return 0;
}

