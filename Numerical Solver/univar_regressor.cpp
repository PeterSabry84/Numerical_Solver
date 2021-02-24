#include "univar_regressor.h"
#include <iostream>
#include <array>
#include <vector>
#include <cmath>

using namespace std;
#include "Linear_system.h"
#include "Matrix.h"

////////////////////////////////////////////////////////
// A class responsible for single variable regression //
////////////////////////////////////////////////////////

univar_regressor::univar_regressor(const double x[], const double y[], const int m, const int s)
	:n(sizeof(x)+1)		//number of points
	, m(m)				//polynomial degree
	, x(new double [n])	//input points
	, y(new double [n]) //target output points
	, valid_solution(false)	//check for ill condition
	, seidel_iterations(0)	//keeps number of iterations of gauss-seidel
{
}


univar_regressor::~univar_regressor()
{
}

/// <summary>
/// Fits a plynomial of a single variable
/// x[]: Array of input points
/// y[]: Array of the output
/// m:	 Polynomial degree
/// n:	 Number of points
/// s:	 Solver choice (1. Gauss Elimination / 2. Gauss-Seidel)
/// </summary>

Matrix univar_regressor::fit(const double x[], const double y[], const int n, const int m, const int s) {

	int i, j;
	double *as = new(double[2 * m + 1]);              
	for (i = 0; i < 2 * m + 1; i++)
	{
		as[i] = 0;
		for (j = 0; j < n; j++)
			as[i] = as[i] + pow(x[j], i);        
	}
	double *C = new double[(m+1)*(m+1)];        //the final coefficients array
	int cntr = 0;
	for (i = 0; i <= m; i++)
		for (j = 0; j <= m; j++)
		{
			C[cntr] = as[i + j];
			cntr++;
		}
			

	double* bs = new double[m + 1];                   
	for (i = 0; i < m + 1; i++)
	{
		bs[i] = 0;
		for (j = 0; j < n; j++)
			bs[i] = bs[i] + pow(x[j], i)*y[j];       
	}

	Matrix A(m + 1, m + 1, C);
	Matrix b(m + 1, 1, bs);
	Matrix out = A.augment(b); //don't forget to multiply with -1 to the b elements.
	cout << "\nAugmented Matrix:\n" << endl;
	cout << out << endl;
	
	// Initialize a linear_system with the augmented matrix [A | b]
	Linear_system S(A, b);

	// Using Gauss Elimination solver
	if (s == 1) {
		Matrix sol = S.solve();
		valid_solution = S.is_valid_solution();
		return sol;
	}
	//Using Gauss-Siedel solver
	else if (s == 2) {
		
		//initialization vector
		double* arr = new double[m+1];
		for (int k = 0; k < m+1; k++)
			arr[k] = 0;

		Matrix sol = S.solve_until(arr, seidel_iterations, 0.001f);
		valid_solution = S.is_valid_solution();
		return sol;
	}
}

/// <summary>
/// Makes a single prediction given:
///		xi:		 single input point
///		coeff[]: coefficients vector (Matrix of size [n x 1]) 
///		m:		 polynomial degree
/// Returns a signle prediction based on the equation:
///		y = a_0 * x^0 + a_1 * x^1 + a_2 x^2 + ..... + a_m x^m
/// </summary>

double univar_regressor::predict(const double xi, const Matrix coeff, const int m)
{
	double yi=0;
	for (int idx = 0; idx <= m; idx++)
		yi += coeff.at(idx, 0) * pow(xi, idx);
	return yi;
}
