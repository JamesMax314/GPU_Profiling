#include <string>
#include <iostream>

#include "../finite_field.hpp"

constexpr int subtest(){

    finite_field<int> a(1388171, 1000);
    finite_field<int> b(1388171, 120000);

    int result = (a-b).value();

    return result;
}
 
int main(){
    std::cout<<"Expected: 1269171, Got: "<<subtest()<<std::endl;
    return 0;
}
