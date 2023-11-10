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
	std::vector<finite_field<int>> vec(size);
	for (int i=0; i<size; i++) {
		vec.at(i) = i+10;
	}


	printf("%i \n", bb(ff).value);
	
	return 0;
}
