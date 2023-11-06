#include <stdio.h>
#include <ctime>
#include <array>
#include <cmath>
#include <numeric>

int main(){
    
    double previous_time;

    const int N_samples = 10000;
    const int N_adds = 1000000;

    double total_time = 0.0;
    double total_time_sq = 0.0;

    int sum = 0;
 
    for(int i =0; i<N_samples; i++){
	std::array<int, N_adds> a, b;
	a.fill(12345678);
	b.fill(87654321);
	previous_time = std::time(NULL);
	for(int j=0; j<N_adds; j++){
	    a.at(j) += b.at(j);
	}
	double time_taken = (std::time(NULL) - previous_time)/double(N_adds);
	total_time += time_taken;
	total_time_sq += time_taken*time_taken;
	
	sum += std::accumulate(a.begin(), a.end(), 0);

    }    

    double average_time = total_time/double(N_samples);    
    double std_dev = std::sqrt((total_time_sq - N_samples*average_time*average_time)/(N_samples-1.0)); 

    printf("Average time to compute: %e s, Standard Deviation: %e s, %d samples \n", average_time, std_dev, N_samples);
    printf("%d", sum);

    return 0;
}
