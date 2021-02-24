#include "multivar_regressor.h"
#include <iostream>
#include <array>
#include <vector>
#include <cmath>
#include <array>
using namespace std;
#include "Linear_system.h"
#include "Matrix.h"

///////////////////////////////////////////////////////
// A class responsible for multi-variable regression //
///////////////////////////////////////////////////////

multivar_regressor::multivar_regressor(const Matrix x, const double y[], const int n, const int m, const int s)
	:n(n)
	, m(m)
	, x(Matrix(m+1, m+1))
	, y(new double[n]) 
	, seidel_iterations(0)
{
}

multivar_regressor::~multivar_regressor()
{

}

/// <summary>
/// Fits a plynomial of a single variable
/// x[]: Array of input points
/// y[]: Array of the output
/// m:	 Input dimension (x1, x2) -> 2 dimensions
/// n:	 Number of points
/// s:	 Solver choice (1. Gauss Elimination / 2. Gauss-Seidel)
/// </summary>

Matrix multivar_regressor::fit(const Matrix x, const double y[], const int n, const int m, const int s) {

	int i, j;
	Matrix A = Matrix(m + 1, m + 1);
	Matrix b = Matrix(m + 1, 1);

	// Populate the augmented matrix as per the algorithm in the book (FIGURE 17.15 pg. 478)
	double sum;
	for (i = 1; i <= m + 1; i++) {

		for (j = 1; j <= i; j++)
		{
			sum = 0;
			for (int l = 0; l < n; l++)
				sum = sum + x.at(i-1,l) * x.at(j-1,l);
			A.set_at(i-1,j-1, sum);
			A.set_at(j-1,i-1, sum);
		}
		sum = 0;
		for (int l = 0; l < n; l++)
			sum = sum + y[l] * x.at(i-1,l);
		b.set_at(i-1,0, sum);

	}
 
	Matrix out = A.augment(b); 
	cout << "\nAugmented Matrix:\n" << endl;
	cout << out << endl;
	
	//Initialize a linear_system with the augmented matrix [A | b]
	Linear_system S(A, b);

	//Using Gauss Elimination solver
	if (s == 1) {
		Matrix sol = S.solve();
		return sol;
	}
	
	//Using Gauss-Seidel solver
	else if (s == 2) {
		//In case of Gauss-Siedel is selected:
		double* arr = new double[m + 1];
		//initialization
		for (int k = 0; k < m + 1; k++)
			arr[k] = 0;
		
		Matrix sol = S.solve_until(arr, seidel_iterations, 0.001f);
		return sol;
	}

}

/// <summary>
/// Makes a single prediction given:
///		xi[]:		 one input vector [x1, x2]
///		coeff[]: coefficients vector (Matrix of size [n x 1]) 
///		m:		  Input dimension
/// Returns a signle prediction based on the equation: >>> considering that x0 = 1 as per the algorithm
///		y = a_0 * x0 + a_1 * x1 + a_2 x2 
/// </summary>
double multivar_regressor::predict(const double xi[], const Matrix coeff, const int m)
{
	double yi = 0;
	for (int idx = 0; idx <= m; idx++)
		yi = yi + coeff.at(idx, 0) * xi[idx];
	return yi;
}
