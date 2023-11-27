#pragma once
#include <vector>
#include <cmath>
#include "../common/cuda_safe_call.hpp"

// template<typename U> U getVal(U rvalue) {return rvalue;};

// template<typename U> U getVal(finite_field<U> rvalue) {return rvalue._value;};

template<typename T> class finite_field{
	private:
	T _value, _prime;

	public:
	__host__ __device__ constexpr finite_field() : _value(0), _prime(2) {}
	__host__ __device__ constexpr finite_field(T prime) : _value(0), _prime(prime) {}
	__host__ __device__ constexpr finite_field(T prime, T value) : _value(value%prime), _prime(prime) {}

	__host__ __device__ constexpr T value() {return this->_value;}
	__host__ __device__ constexpr T prime() {return this->_prime;}

	template<typename U> __host__ __device__ finite_field<T> pow(U exponent){
		T newVal = std::pow(_value, exponent);
		return finite_field<T>(_prime, newVal);
	}

	template<typename U> __host__ __device__ finite_field<T> pow(finite_field<U> rvalue){
		U exponent = rvalue.value();
		T newVal = std::pow(_value, exponent);
		return finite_field<T>(_prime, newVal);
	}

	template<typename U> __host__ __device__ finite_field<T>& operator+=(U rvalue){
		this->_value = (_value + rvalue)%_prime;
		return *this;
	}

	template<typename U> __host__ __device__ finite_field<T>& operator+=(finite_field<U> rvalue){
		this->_value = (_value + rvalue.value())%_prime;
		return *this;
	}

	template<typename U> __host__ __device__ finite_field<T>& operator-=(U rvalue){
		this->_value = (_value - rvalue + _prime)%_prime;
		return *this;
	}

	template<typename U> __host__ __device__ constexpr finite_field<T>& operator-=(finite_field<U> rvalue){
		this->_value = (_value + (_prime - rvalue.value()))%_prime;
		return *this;
	}

	template<typename U> __host__ __device__ finite_field<T>& operator*=(U rvalue){
		this->_value = (_value * rvalue)%_prime;
		return *this;
	}

	template<typename U> __host__ __device__ finite_field<T>& operator*=(finite_field<U> rvalue){
		this->_value = (_value * rvalue.value())%_prime;
		return *this;
	}

	template<typename U> __host__ __device__ finite_field<T>& operator=(U rvalue){
		_value = (rvalue)%_prime;
		return *this;}

	template<typename U> __host__ __device__ finite_field<T>& operator=(finite_field<U> rvalue){
		_value = (rvalue.value())%_prime;
		return *this;}
};

//Define operators for primites/other types that allows for e.g. int + finite_field = finite_field
template<typename T> inline __host__ __device__ finite_field<T> operator+(finite_field<T> lvalue, finite_field<T> rvalue){
		finite_field<T> result(lvalue.prime());
		result += rvalue;
		return result;
}
template<typename T> inline __host__ __device__ finite_field<T> operator+(finite_field<T> lvalue, T rvalue){return finite_field<T>(lvalue.prime(), (lvalue.value() + rvalue));}
template<typename T> inline __host__ __device__ finite_field<T> operator+(T lvalue, finite_field<T> rvalue){return finite_field<T>(rvalue.prime(), (lvalue + rvalue.value()));}

template<typename T> inline __host__ __device__ constexpr finite_field<T> operator-(finite_field<T> lvalue, finite_field<T> rvalue){

	finite_field<T> result = lvalue;
	result -= rvalue;
	return result;
}
template<typename T> inline __host__ __device__ finite_field<T> operator-(finite_field<T> lvalue, T rvalue){return finite_field<T>(lvalue.prime(), (lvalue.value() - rvalue));}
template<typename T> inline __host__ __device__ finite_field<T> operator-(T lvalue, finite_field<T> rvalue){return finite_field<T>(rvalue.prime(), (lvalue - rvalue.value()));}

template<typename T> inline __host__ __device__ finite_field<T> operator*(finite_field<T> lvalue, finite_field<T> rvalue){
	finite_field<T> result(lvalue.prime());
	result *= rvalue;
	return result;
}
template<typename T> inline __host__ __device__ finite_field<T> operator*(finite_field<T> lvalue, T rvalue){return finite_field<T>(lvalue.prime(), (lvalue.value() * rvalue));}
template<typename T> inline __host__ __device__ finite_field<T> operator*(T lvalue, finite_field<T> rvalue){return finite_field<T>(rvalue.prime(), (lvalue * rvalue.value()));}

template<typename T> inline __host__ __device__ finite_field<T> operator/(finite_field<T> lvalue, finite_field<T> rvalue){
	finite_field<T> result(lvalue.prime());
	result /= rvalue;
	return result;
}
template<typename T> inline __host__ __device__ finite_field<T> operator/(finite_field<T> lvalue, T rvalue){return finite_field<T> (lvalue.prime(), (lvalue.value() / rvalue));}
template<typename T> inline __host__ __device__ finite_field<T> operator/(T lvalue, finite_field<T> rvalue){return finite_field<T> (rvalue.prime(), (lvalue / rvalue.value()));}


enum ComputeMethod {
	CPU,
	GPU
};

class BlackBox {
	private:
	int _degree;

	public:
	BlackBox() : _degree(1) { }
	BlackBox(int degree) : _degree(degree) { }

	int degree() {return this->_degree;}
	void set_degree(int degree) {this->_degree = degree;}

	template<typename T>
	__host__ __device__ finite_field<T> polynomial(finite_field<T>& rvalue){
		finite_field<T> result(rvalue.prime());
		for (int i=0; i<_degree; i++) {
			result += i*rvalue.pow(i/_degree)/8;
		}
		return result;
	}

	template<typename T>
	__host__ __device__ finite_field<T> operator()(finite_field<T>& rvalue) 
	{
		finite_field<T> result(7); //Member of Z_7
		result = polynomial(rvalue);
		return result;
	};
};