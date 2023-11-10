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
	std::vector<finite_field<int>> vec;
	vec.emplace_back(ff);
	vec.emplace_back(ff2);

	printf("%i \n", bb(vec).value);
	
	return 0;
}
