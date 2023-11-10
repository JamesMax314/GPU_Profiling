#include <vector>
#include <cmath>

// template<typename U> U getVal(U rvalue) {return rvalue;};

// template<typename U> U getVal(finite_field<U> rvalue) {return rvalue._value;};

template<typename T> class finite_field{
	private:
	T _value, _prime;

	public:
	finite_field(T prime, T value) : _value(value%prime), _prime(prime) {}

	T value() {return this->_value;}
	T prime() {return this->_prime;}

	template<typename U> finite_field<T> operator+(U rvalue){return finite_field(_prime, (_value + rvalue));}
	template<typename U> finite_field<T> operator+(finite_field<U> rvalue){return finite_field(_prime, (_value + rvalue._value()));}

	template<typename U> finite_field<T> operator-(U rvalue){return finite_field(_prime, (_value - rvalue));}
	template<typename U> finite_field<T> operator-(finite_field<U> rvalue){return finite_field(_prime, (_value - rvalue._value()));}

	template<typename U> finite_field<T> operator*(U rvalue){return finite_field(_prime, (_value * rvalue));}
	template<typename U> finite_field<T> operator*(finite_field<U> rvalue){return finite_field(_prime, (_value * rvalue._value()));}

	template<typename U> finite_field<T> operator/(U rvalue){return finite_field(_prime, (_value / rvalue));}
	template<typename U> finite_field<T> operator/(finite_field<U> rvalue){return finite_field(_prime, (_value / rvalue._value()));}

	template<typename U> finite_field<T> pow(U rvalue){
		T newVal = _value;
		for (int i=0; i<rvalue-1; i++) {
			newVal *= _value;
		}
		return finite_field(_prime, newVal);
	}

	template<typename U> finite_field<T>& operator+=(U rvalue){
		this->_value = (_value + rvalue.value())%_prime;
		return *this;
	}

	template<typename U> finite_field<T>& operator-=(U rvalue){
		this->_value = (_value - rvalue)%_prime;
		return *this;
	}

	template<typename U> finite_field<T>& operator*=(U rvalue){
		this->_value = (_value * rvalue)%_prime;
		return *this;
	}

	template<typename U> finite_field<T>& operator=(U rvalue){
		_value = (rvalue)%_prime;
		return *this;}

};

class black_box {
	private:

	public:
	black_box() {};

	// Called by cuda transform to do operation at many different values
	template<typename T>

	// finite_field<T> operator()(finite_field<T>& rvalue) 
	// {
	// 	finite_field<T> result(rvalue.prime(), 0);
	// 	for (int i=1; i<100; i++) {
	// 		result += rvalue.pow(i)*i;
	// 	}
	// 	return rvalue.pow(7);
	// };

	finite_field<T> operator()(finite_field<T>& rvalue) 
	{
		finite_field<T> result(rvalue.prime(), 0);
		result = rvalue;
		return rvalue.pow(7);
	};
};
