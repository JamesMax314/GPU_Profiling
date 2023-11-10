#include <cstdio>
#include "finite_field.hpp"

finite_field<int> ff(13, 2);

int main(){

	for(int i=0; i<100; i++){
		ff += 1;
		printf("%i \n", ff.value);
	}
	
	return 0;
}
