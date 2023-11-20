#include "../finite_field.hpp"

finite_field<int> a(1388171, 1000);
finite_field<int> b(1388171, 120000);

static_assert((a-b) == finite_field<int>(1388171, 1269171));

