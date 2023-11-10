#include <cstdio>
#include <algorithm> 
#include "finite_field.hpp"

finite_field<int> ff(100, 2);
finite_field<int> ff2(13, 12);
black_box bb;

int main(){

	// for(int i=0; i<100; i++){
	// 	ff += 1;
	// 	printf("%i \n", ff.value);
	// }
	int size = 10000;
	std::vector<finite_field<int>> vec;
	std::vector<finite_field<int>> vecOut;
	for (int i=0; i<size; i++) {
		vec.emplace_back(13, i+10);
		vecOut.emplace_back(13, 0);
	}

	std::transform(vec.begin(), vec.end(), vecOut.begin(), bb);
	
	for (int i=0; i<size; i++) {
		printf("%d \n", vecOut[i]);
	}

	// printf("%i \n", bb(ff).value);
	
	return 0;
}
