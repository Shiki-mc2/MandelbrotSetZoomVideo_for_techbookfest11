#include <iostream>
#include <mpirxx.h>
#include <complex>
#include <time.h>
#include <fstream>
#include <iomanip>
#include <string>

int main(int argc, char* argv[], char* envp[])
{
    int row = std::stoi(argv[1]);
    int col = std::stoi(argv[2]);
    int prec = std::stoi(argv[3]);
    mpf_class xmin(argv[4], prec);
    mpf_class ymax(argv[5], prec);
    mpf_class dpp(argv[6], prec);

    int   N = std::stoi(argv[7]);
    float R = std::stof(argv[8]);
    float R2 = R * R;

    mpf_class c_imag(0.0, prec);
    mpf_class c_real(0.0, prec);
    mpf_class z_real(0.0, prec);
    mpf_class z_imag(0.0, prec);
    mpf_class z_real_tmp(0.0, prec);

    std::string fname = argv[9];
    std::ofstream w;

    w.open(fname, std::ios::out);
    for (int i = 0; i < row; i++) {
        c_imag = ymax - dpp * i;
        for (int j = 0; j < col; j++) {
            z_real = 0.0;
            z_imag = 0.0;
            z_real_tmp = 0.0;
            c_real = xmin + dpp * j;
            int k = 0;
            for (k = 0; k < N; k++) {
                z_real_tmp = z_real * z_real - z_imag * z_imag + c_real;
                z_imag = 2 * z_real * z_imag + c_imag;
                z_real = z_real_tmp;
                if (z_real * z_real + z_imag * z_imag > R2) {
                    break;
                }
            }
            if (j != 0) {
                w << ",";
            }
            w << k;
        }
        w << std::endl;
    }
    w.close();
}