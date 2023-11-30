#include <string>
#include <iostream>
#define CATCH_CONFIG_MAIN
#include "catch.hpp"

#include "../finite_field.hpp"

constexpr int subtest(){

    finite_field<int> a(1388171, 1000);
    finite_field<int> b(1388171, 120000);

    int result = (a-b).value();

    return result;
}

TEST_CASE( "Finite Field Subtraction" ) {
    finite_field<int> a(1388171, 1000);
    finite_field<int> b(1388171, 120000);

    REQUIRE( (a-b).value() == 1269171 );
}

TEST_CASE( "Finite Field Multiplication" ) {
    finite_field<unsigned int> a(1388171, 1000);
    finite_field<unsigned int> b(1388171, 120000);
    unsigned int multiplier = 120000;
    // std::cout << (a).value() << std::endl;

    REQUIRE( (a*b).value() == 617294 );
    REQUIRE( (a*multiplier).value() == 617294 );
    REQUIRE( (multiplier*a).value() == 617294 );
    REQUIRE( (a*=b).value() == 617294 );
    REQUIRE( (b*=multiplier).value() == 502217 );
}
 
// int main(){
//     std::cout<<"Expected: 1269171, Got: "<<subtest()<<std::endl;
//     return 0;
// }
