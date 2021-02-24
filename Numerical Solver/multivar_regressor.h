#pragma once
#include <array>
#include "Matrix.h"

class multivar_regressor
{
private:
	int n, m;
	double *y;
	Matrix  x;

public:
	int seidel_iterations;
	multivar_regressor(const Matrix x, const double y[], const int n,const int m, const int s);
	~multivar_regressor();

	Matrix fit(const Matrix x, const double y[], const int n, const int m, const int s);
	double predict(const double x[], const Matrix coeff, const int m);
};

