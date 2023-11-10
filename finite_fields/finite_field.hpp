template<typename T> class finite_field{

	public:
	finite_field(T prime, T value) : value(value), prime(prime) {}

	template<typename U> finite_field<T> operator+(U rvalue){return finite_field(prime, (value + rvalue)%prime);}

	template<typename U> finite_field<T>& operator+=(U rvalue){
			this->value = (value + rvalue)%prime;
			return *this;
	}

	template<typename U> finite_field<T> operator-(U rvalue){return finite_field(prime, (value - rvalue)%prime);}

	template<typename U> finite_field<T>& operator-=(U rvalue){
			this->value = (value - rvalue)%prime;
			return *this;
	}

	template<typename U> finite_field<T> operator=(U rvalue){value = (rvalue)%prime;}

	T value, prime;
};

