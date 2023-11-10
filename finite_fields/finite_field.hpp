#include <vector>
#include <cmath>

// template<typename U> U getVal(U rvalue) {return rvalue;};

// template<typename U> U getVal(finite_field<U> rvalue) {return rvalue._value;};

template<typename T> class finite_field{
	private:
	T _value, _prime;

	public:
	finite_field() : _value(0), _prime(2) {}
	finite_field(T prime) : _value(0), _prime(prime) {}
	finite_field(T prime, T value) : _value(value%prime), _prime(prime) {}

	T value() {return this->_value;}
	T prime() {return this->_prime;}

	template<typename U> finite_field<T> pow(U rvalue){
		T newVal = _value;
		for (int i=0; i<rvalue-1; i++) {
			newVal *= _value;
		}
		return finite_field(_prime, newVal);
	}

	template<typename U> finite_field<T> pow(finite_field<U> rvalue){

		U exponent = rvalue.value();

		T newVal = std::pow(_value, exponent);

		
		return finite_field(_prime, newVal);
	}

	template<typename U> finite_field<T>& operator+=(U rvalue){
		this->_value = (_value + rvalue)%_prime;
		return *this;
	}

	template<typename U> finite_field<T>& operator+=(finite_field<U> rvalue){
		this->_value = (_value + rvalue.value())%_prime;
		return *this;
	}

	template<typename U> finite_field<T>& operator-=(U rvalue){
		this->_value = (_value - rvalue)%_prime;
		return *this;
	}

	template<typename U> finite_field<T>& operator-=(finite_field<U> rvalue){
		this->_value = (_value - rvalue.value())%_prime;
		return *this;
	}

	template<typename U> finite_field<T>& operator*=(U rvalue){
		this->_value = (_value * rvalue)%_prime;
		return *this;
	}

	template<typename U> finite_field<T>& operator*=(finite_field<U> rvalue){
		this->_value = (_value * rvalue.value())%_prime;
		return *this;
	}

	template<typename U> finite_field<T>& operator=(U rvalue){
		_value = (rvalue)%_prime;
		return *this;}

	template<typename U> finite_field<T>& operator=(finite_field<U> rvalue){
		_value = (rvalue.value())%_prime;
		return *this;}
};

//Define operators for primites/other types that allows for e.g. int + finite_field = finite_field
template<typename T> inline finite_field<T> operator+(finite_field<T> lvalue, finite_field<T> rvalue){
		finite_field result(lvalue.prime());
		result += rvalue;
		return result;
}
template<typename T> inline finite_field<T> operator+(finite_field<T> lvalue, T rvalue){return finite_field(lvalue.prime(), (lvalue.value() + rvalue));}
template<typename T> inline finite_field<T> operator+(T lvalue, finite_field<T> rvalue){return finite_field(rvalue.prime(), (lvalue + rvalue.value()));}

template<typename T> inline finite_field<T> operator-(finite_field<T> lvalue, finite_field<T> rvalue){
	finite_field result(lvalue.prime());
	result -= rvalue;
	return result;
}
template<typename T> inline finite_field<T> operator-(finite_field<T> lvalue, T rvalue){return finite_field(lvalue.prime(), (lvalue.value() - rvalue));}
template<typename T> inline finite_field<T> operator-(T lvalue, finite_field<T> rvalue){return finite_field(rvalue.prime(), (lvalue - rvalue.value()));}

template<typename T> inline finite_field<T> operator*(finite_field<T> lvalue, finite_field<T> rvalue){
	finite_field result(lvalue.prime());
	result *= rvalue;
	return result;
}
template<typename T> inline finite_field<T> operator*(finite_field<T> lvalue, T rvalue){return finite_field(lvalue.prime(), (lvalue.value() * rvalue));}
template<typename T> inline finite_field<T> operator*(T lvalue, finite_field<T> rvalue){return finite_field(rvalue.prime(), (lvalue * rvalue.value()));}

template<typename T> inline finite_field<T> operator/(finite_field<T> lvalue, finite_field<T> rvalue){
	finite_field result(lvalue.prime());
	result /= rvalue;
	return result;
}
template<typename T> inline finite_field<T> operator/(finite_field<T> lvalue, T rvalue){return finite_field(lvalue.prime(), (lvalue.value() / rvalue));}
template<typename T> inline finite_field<T> operator/(T lvalue, finite_field<T> rvalue){return finite_field(rvalue.prime(), (lvalue / rvalue.value()));}

class black_box {
	private:

	public:
	black_box() {};

	// Called by cuda transform to do operation at many different values

	// finite_field<T> operator()(finite_field<T>& rvalue) 
	// {
	// 	finite_field<T> result(rvalue.prime(), 0);
	// 	for (int i=1; i<100; i++) {
	// 		result += rvalue.pow(i)*i;
	// 	}
	// 	return rvalue.pow(7);
	// };
	
	template<typename T>
	finite_field<T> polynomial(finite_field<T>& rvalue){
		finite_field<T> result = rvalue + 5*rvalue.pow(3) + 10;
		return result;
	}

	template<typename T>
	finite_field<T> operator()(finite_field<T>& rvalue) 
	{
		finite_field<T> result(7); //Member of Z_7
		result = polynomial(rvalue);
		return result;
	};
};
