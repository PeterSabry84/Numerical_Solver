#pragma once
#include <array>
#include "Matrix.h"
class Linear_system
{
private:
	bool valid_solution;
	const int n; 
	const int m;
	Matrix A;
	double* x;

public:
	//default constructor
	Linear_system(const int& x, const int& y);
	Linear_system(const int& x, const int& y, const double a[]);
	Linear_system(const int& x, const int& y, const Matrix& A);
	Linear_system(const Matrix& A, const Matrix& b);
	~Linear_system();
	
	Matrix solve();
	Matrix solve_until(double initials[], int & n_iter, float epsilon) const;
	bool is_valid_solution() { return valid_solution;}

};

