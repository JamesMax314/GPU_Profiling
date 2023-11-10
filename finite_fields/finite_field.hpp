#include <vector>
#include <cmath>

template<typename T> class finite_field{

	public:
	finite_field(T prime, T value) : value(value%prime), prime(prime) {}

	template<typename U> U getVal(U rvalue) {return rvalue;};
	template<typename U> U getVal(finite_field<U> rvalue) {return rvalue.value;};

	template<typename U> finite_field<T> operator+(U rvalue){return finite_field(prime, (value + getVal(rvalue)));}
	template<typename U> finite_field<T> operator-(U rvalue){return finite_field(prime, (value - getVal(rvalue)));}
	template<typename U> finite_field<T> operator*(U rvalue){return finite_field(prime, (value * getVal(rvalue)));}
	template<typename U> finite_field<T> operator/(U rvalue){return finite_field(prime, (value / getVal(rvalue)));}

	template<typename U> finite_field<T> pow(U rvalue){
		T newVal = value;
		for (int i=0; i<getVal(rvalue)-1; i++) {
			newVal *= value;
		}
		return finite_field(prime, newVal);
	}

	template<typename U> finite_field<T>& operator+=(U rvalue){
		this->value = (value + getVal(rvalue))%prime;
		return *this;
	}

	template<typename U> finite_field<T>& operator-=(U rvalue){
		this->value = (value - getVal(rvalue))%prime;
		return *this;
	}

	template<typename U> finite_field<T>& operator*=(U rvalue){
		this->value = (value * getVal(rvalue))%prime;
		return *this;
	}

	template<typename U> finite_field<T> operator=(U rvalue){value = (getVal(rvalue))%prime;}

	T value, prime;
};


class black_box {
	private:

	public:
	black_box() {};

	// Called by cuda transform to do operation at many different values
	template<typename T>
	finite_field<T> operator()(std::vector<finite_field<T>> rvalues) {return rvalues[0].pow(7) + rvalues[1].pow(8);};
};