#include "utils.hpp"

std::string files::generateRandomCode(int length) {
    static const char alphanum[] =
        "0123456789"
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        "abcdefghijklmnopqrstuvwxyz";

    std::string code;
    
    // Seed the random number generator
    std::srand(static_cast<unsigned int>(std::time(nullptr)));

    for (int i = 0; i < length; ++i) {
        code += alphanum[std::rand() % (sizeof(alphanum) - 1)];
    }

    return code;
}

void files::saveArr2D(std::vector<std::vector<double>> arr, std::string file) {
    std::ofstream myfile(file);
    if (myfile.is_open()) {
        for (int k = 0; k < static_cast<int>(arr.size()); k++) {
            for (int i = 0; i < static_cast<int>(arr[0].size()); i++) {
                if (i < static_cast<int>(arr[0].size())-1)
                    myfile << arr[k][i] << " ";
                else
                    myfile << arr[k][i];
            }
            myfile << "\n";
        }
        myfile.close();
        printf("Saved to file: %s \n", file.c_str());
    } else {
        printf("no file \n");
    }
}

void files::saveVec1D(std::vector<double> vec, std::string file) {
    std::ofstream myfile(file);
    if (myfile.is_open()) {
        for (int k = 0; k < static_cast<int>(vec.size()); k++) {
            myfile << vec[k];
            myfile << "\n";
        }
        myfile.close();
        printf("Saved to file: %s \n", file.c_str());
    } else {
        printf("no file \n");
    }
}