#pragma once
#include <math.h>
#include <vector>
#include <random>
#include <fstream>
#include <iostream>
#include <ctime>

namespace files
{
    void saveArr2D(std::vector<std::vector<double>> arr, std::string file);
    void saveVec1D(std::vector<double> vec, std::string file);
    std::string generateRandomCode(int length);
}

