// Numerical Solver.cpp : This file contains the 'main' function. Program execution begins and ends there.
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
using namespace std;
#include "Linear_system.h"
#include "Matrix.h"
#include "univar_regressor.h"
#include "multivar_regressor.h"
#include "Newton_interpolator.h"
#include "Spline_Interpolator.h"

#include <cmath>

#include "gnuplot_i.hpp" //Gnuplot class handles POSIX-Pipe-communikation with Gnuplot


//Normalization functions >>> Used for normalizing housing data 

Matrix norm_x(const Matrix x, const int n)
{
	int i, j;
	Matrix x_tmp = Matrix(3, n);
	for (i = 0; i < n; i++)
		x_tmp.set_at(0, i, 1);
	for (i = 1; i < 3; i++)
	{
		double x_min = x.at(i, 0);
		double x_max = x.at(i, 0);
		for (int j = 1; j < n; j++) {
			if (x.at(i, j) < x_min) {
				x_min = x.at(i, j);
			}
			if (x.at(i, j) > x_max) {
				x_max = x.at(i, j);
			}
		}
		for (int j = 0; j < n; j++)
			x_tmp.set_at(i, j, (x.at(i, j) - x_min) / (x_max - x_min));
	}

	return x_tmp;
}
double* norm_y(const double y[], const int n) {
	int i;
	double* y_norm = new double[n];
	double y_min = y[0];
	double y_max = y[0];
	for (i = 0; i < n; i++) {
		if (y[i] < y_min)
			y_min = y[i];
		if (y[i] > y_max)
			y_max = y[i];
	}
	for (i = 0; i < n; i++)
		y_norm[i] = (y[i] - y_min) / (y_max - y_min);
	return y_norm;
}

double denorm_y(const double y_norm, const double y_min, const double y_max) {
	double y_denorm = y_norm * (y_max - y_min) + y_min;
	return y_denorm;
}

double calc_mse(const double y[], const double y_pred[], const int n) {
	double mse = 0;
	for (int i = 0; i < n; i++)
		mse = mse + pow((y[i] - y_pred[i]),2) /n;
	return mse;
}



void regress_line(string filename, int m, int s) {
	int i, n;
	// Reading training data file

	ifstream  trainData;
	trainData.open(filename);
	cout << "Opened file: " << filename;


	string line;
	vector<vector<string> > parsedCsv;
	while (getline(trainData, line))
	{
		stringstream lineStream(line);
		string cell;
		vector<string> parsedRow;
		while (getline(lineStream, cell, ','))
		{
			parsedRow.push_back(cell);
		}

		parsedCsv.push_back(parsedRow);
	}
	n = parsedCsv.size() - 1;
	double *x = new double[n];
	double *y = new double[n];

	for (size_t i = 1; i <= n; ++i) // start at i = 1 to skip header line
	{
		x[i - 1] = stod(parsedCsv[i][0]);
		y[i - 1] = stod(parsedCsv[i][1]);
	}



	univar_regressor polyRegress = univar_regressor(x, y, m, s);
	Matrix coeff = polyRegress.fit(x, y, n, m, s);

	bool valid_sol = polyRegress.is_valid_solution();
	int num_iterations = polyRegress.seidel_iterations;

	if (valid_sol)
		cout << "System has valid Solution" << endl;
	else
		cout << "System is ill conditioned" << endl;

	cout << "\nThe values of the solution coefficients are:\n";
	for (int idx = 0; idx <= m; idx++)
		cout << "x^" << idx << "=" << coeff.at(idx, 0) << endl;            // Print the values of x^0,x^1,x^2,x^3,....    
	cout << "\nThe fitted Polynomial is given by:\ny=";
	for (int idx = 0; idx <= m; idx++)
		cout << " + (" << coeff.at(idx, 0) << ")" << "x^" << idx;
	cout << "\n";
	if (s == 2)
		cout << "\n\t Gauss Seidel Iterations: " << num_iterations << endl;

	// Running Prediction on training data
	double *y_pred = new double[n];
	for (i = 0; i < n; i++)
		y_pred[i] = polyRegress.predict(x[i], coeff, m);

	// Printing Actual vs. prediction on console
	cout << "\n Predictions: \n" << endl;
	cout << "\nx   y_pred    y" << endl;
	cout << "=================" << endl;


	for (i = 0; i < n; i++) {
		cout << x[i] << " " << y_pred[i] << " " << y[i] << endl;
	}


	//Plotting original points vs. fitted line
	std::vector<double> x_vec(x, x + n);
	std::vector<double> y_vec(y, y + n);
	

	//Formulate the equation with the coefficients
	string eqn = "";
	for (i = 0; i < m + 1; i++)
		eqn = eqn + "+" + to_string(coeff.at(i, 0)) + "*(x**" + to_string(i) + ")";

	//Use GNUPLOT
	Gnuplot g1("lines");
	g1.set_grid();
	g1.plot_equation(eqn, "Fitted Line");  //plot the fitted line
	g1.set_style("points").plot_xy(x_vec, y_vec, "Original Points"); //plot the original points
	
}

int regress_2d(const int x1, const int x2, const int s) {  //Accepts the indexes of the needed two columns for regression
	int i, j, n, m;
	m = 2;    // two dimensional input x: x_1, x_2
	ifstream  trainData;
	try {
		trainData.open("Housing.csv");
		cout << "Opened file: Housing.csv";
	}
	catch (const char* cstr) {
		cerr << cstr << '\n';
		exit(1);
	}
	string line;
	vector<vector<string> > parsedCsv;
	while (getline(trainData, line))
	{
		stringstream lineStream(line);
		string cell;
		vector<string> parsedRow;
		while (getline(lineStream, cell, ','))
		{
			parsedRow.push_back(cell);
		}

		parsedCsv.push_back(parsedRow);
	}
	n = parsedCsv.size() - 1;

	Matrix x = Matrix(3, n);
	double *y = new double[n];

	for (i = 1; i <= n; ++i) // start at i = 1 to skip header line
	{
		x.set_at(0, i - 1, 1);        // put 1's in x0 for the algorithm to work, data points will be stored in x1, x2
		x.set_at(1, i - 1, stod(parsedCsv[i][x1]));
		x.set_at(2, i - 1, stod(parsedCsv[i][x2]));
		y[i - 1] = stod(parsedCsv[i][4]);
	}


	// Traget column "Price" is in a very large scale
	//Normalizing "Price"
	double *y_norm = new double[n];
	y_norm = norm_y(y, n);

	//Normalizing input features
	Matrix x_norm(m + 1, m + 1);
	x_norm = norm_x(x, n);


	multivar_regressor polyRegress2D = multivar_regressor(x_norm, y_norm, n, m, s);
	Matrix coeff = polyRegress2D.fit(x_norm, y_norm, n, m, s);

	cout << "\nThe values of the solution coefficients are:\n";
	for (int idx = 0; idx <= m; idx++)
		cout << "x_" << idx << "=" << coeff.at(idx, 0) << endl;            // Print the values of x^0,x^1,x^2,x^3,....    
	cout << "\nThe fitted Polynomial is given by:\ny=";
	for (int idx = 0; idx <= m; idx++)
		cout << " + (" << coeff.at(idx, 0) << ")" << "x_" << idx;
	cout << "\n";

	//Getting Min, Max price for de-normalizing prediction
	double y_min = y[0];
	double y_max = y[0];
	for (i = 0; i < n; i++) {
		if (y[i] < y_min)
			y_min = y[i];
		if (y[i] > y_max)
			y_max = y[i];
	}

	// Prediction
	double* xi = new double[3];
	double *y_pred = new double[n];
	double* y_norm_pred = new double[n];
	for (i = 0; i < n; i++)
	{
		for (j = 0; j < 3; j++)
			xi[j] = x_norm.at(j, i);
		y_norm_pred[i] = polyRegress2D.predict(xi, coeff, m);
	}

	for (i = 0; i < n; i++)
	{
		y_pred[i] = denorm_y(y_norm_pred[i], y_min, y_max);
	}

	double mse = calc_mse(y_norm, y_norm_pred,n);
	cout << "\n\n=========================================" << endl;
	cout << "Mean Square Error= " << mse << endl;
	cout << "\n\n=========================================" << endl;

	cout << "\n Sample Predictions: \n" << endl;
	cout << "\nx1 x2  Pred_Price Price" << endl;
	cout << "=========================================" << endl;

	for (i = 0; i < 11; i++) {
		cout << x.at(1, i) << "  " << x.at(2, i) << "       " << y_pred[i] << "     " << y[i] << endl;
	}
	cout << "=========================================" << endl;
	cout << "=========================================" << endl;
	return 0;
}


void interpolate(string filename) {

	//Read input file:

	int i, n;
	// Reading training data file

	ifstream  trainData;
	trainData.open(filename);
	cout << "Opened file: " << filename;


	string line;
	vector<vector<string> > parsedCsv;
	while (getline(trainData, line))
	{
		stringstream lineStream(line);
		string cell;
		vector<string> parsedRow;
		while (getline(lineStream, cell, ','))
		{
			parsedRow.push_back(cell);
		}

		parsedCsv.push_back(parsedRow);
	}
	n = parsedCsv.size() - 1;
	double *x = new double[n];
	double *y = new double[n];

	for (size_t i = 1; i <= n; ++i) // start at i = 1 to skip header line
	{
		x[i - 1] = stod(parsedCsv[i][0]);
		y[i - 1] = stod(parsedCsv[i][1]);
	}

	cout << "\nNewton Interpolation" << endl;
	cout << "=======================" << endl;

	//Define interpolation
	Newton_interpolator ni(x, y, n);
	//Fit to get coefficients
	double* ni_coeff = new double[n];
	ni_coeff = ni.fit();
	//Output to file
	string coeff_filename = "newton_coeff_" + filename;
	ofstream outfile;
	outfile.open(coeff_filename);
	for (i = 0; i < n; i++)
		outfile << ni_coeff[i] << endl;
	outfile.close();


	//doubling the points x and y.
	double* x_ = new double[2 * n - 1];
	double* y_ = new double[2 * n - 1];
	double* x__ = new double[(n + (n - 1)) + (2 * n - 2)];
	double* y__ = new double[(n + (n - 1)) + (2 * n - 2)];
	for (int i = 0; i < n + (n - 1); i++)
	{
		if (i % 2 != 1)
		{
			x_[i] = x[i / 2];
			y_[i] = y[i / 2];
		}
		else
		{
			x_[i] = (x[i / 2 + 1] + x[i / 2]) / 2.0;
			y_[i] = ni.interpolate(x_[i]);
		}


	}

	//doubling the points once more

	for (int i = 0; i < (n + (n - 1)) + (2 * n - 2); i++)
	{
		if (i % 2 != 1)
		{
			x__[i] = x_[i / 2];
			y__[i] = y_[i / 2];
		}
		else
		{
			x__[i] = (x_[i / 2 + 1] + x_[i / 2]) / 2.0;
			y__[i] = ni.interpolate(x__[i]);
		}


	}


	//Output to file
	ofstream outfile1;
	outfile1.open("newton_doublePoints_" + filename);
	for (i = 0; i < 2 * n - 1; i++)
		outfile1 << x_[i] << "," << y_[i] << endl;
	outfile1.close();

	ofstream outfile2;
	outfile2.open("newton_FourPoints_" + filename);
	for (i = 0; i < (n + (n - 1)) + (2 * n - 2); i++)
		outfile2 << x__[i] << "," << y__[i] << endl;
	outfile2.close();

	////////////////////////////////////////////////////////
	// spline interpolation:
	///////////////////////////////////////////////////////
	cout << "\nSpline Interpolation: " << endl;
	cout << "=======================" << endl;

	//Define interpolation

	Spline_interpolator s(x, y, n);

	//Fit to get coefficients
	s.fit();

	Matrix s_coeff = s.get_coefficients();

	cout << s_coeff;
	cout << endl;

	//Output to file
	string s_coeff_filename = "spline_coeff_" + filename;
	ofstream s_outfile;
	outfile.open(coeff_filename);
	outfile << s_coeff;
 
	outfile.close();
	
	//doubling the points x and y.
	//double* x_ = new double[2 * n - 1];
	//double* y_ = new double[2 * n - 1];
	//double* x__ = new double[(n + (n - 1)) + (2 * n - 2)];
	//double* y__ = new double[(n + (n - 1)) + (2 * n - 2)];
	for (int i = 0; i < n + (n - 1); i++)
	{
		if (i % 2 != 1)
		{
			x_[i] = x[i / 2];
			y_[i] = y[i / 2];
		}
		else
		{
			x_[i] = (x[i / 2 + 1] + x[i / 2]) / 2.0;
			y_[i] = s.interpolate(x_[i]);
		}


	}

	//doubling the points once more

	for (int i = 0; i < (n + (n - 1)) + (2 * n - 2); i++)
	{
		if (i % 2 != 1)
		{
			x__[i] = x_[i / 2];
			y__[i] = y_[i / 2];
		}
		else
		{
			x__[i] = (x_[i / 2 + 1] + x_[i / 2]) / 2.0;
			y__[i] = s.interpolate(x__[i]);
		}


	}


	//Output to file
	//ofstream outfile1;
	outfile1.open("Spline_doublePoints_" + filename);
	for (i = 0; i < 2 * n - 1; i++)
		outfile1 << x_[i] << "," << y_[i] << endl;
	outfile1.close();

	//ofstream outfile2;
	outfile2.open("Spline_FourPoints_" + filename);
	for (i = 0; i < (n + (n - 1)) + (2 * n - 2); i++)
		outfile2 << x__[i] << "," << y__[i] << endl;
	outfile2.close();


}



int main()
{
	int number = 0;
	cout << "Chose what do you want to do: \n \t0: Gauss-Seidel test. \n \t1: Single variable Polynomial Regression. \n\t2: 2D Ploynomial Regression. \n\t3: Interpolation. \n";
	cin >> number;

	if (number == 0)
	{
		double temp_[12] = { 3,-0.1, -0.2, 7.85, 0.1, 7, -0.3, -19.3, 0.3, -0.2, 10, 71.4 };
		Matrix toto(3, 4, temp_);
		Linear_system soso(3, 4, toto);
		cout << "\nUsing Gauss elimination:\n" << soso.solve() << endl;
		double initials[3] = { 0, 0, 0 };
		cout << "\nUsing Seidel elimination:\n" << soso.solve_iteratively(initials, 100) << endl;
		int n_ = 0;
		cout << "\nUsing Seidel elimination with solve_until:\n" << soso.solve_until(initials, n_, 0.01) << endl;
		cout << "\nNumber of iterations: " << n_ << endl;
	}

	else if (number == 1)
	{
		int m;
		cout << "Linear Regression" << endl;
		cout << "\nWhat degree of Polynomial do you want to use for the fit?\n";
		cin >> m;

		cout << "\n Chose to Sove using: \n\t1.Gauss Scaled Elimination. \n\t2.Iterative Gauss-Siedel\n";
		int s;
		cin >> s;

		cout << "===========================" << endl;
			cout << "\n\nUsing First Dataset:" << endl;
			cout << "===========================" << endl;

			regress_line("2_a_dataset_1.csv", m, s);

			cout << "\n===========================" << endl;
			cout << "\n\nUsing Second Dataset:" << endl;
			cout << "===========================" << endl;

			regress_line("2_a_dataset_2.csv", m, s);
	

	}
	else if (number == 2) {

		cout << "\n Chose to Sove using: \n\t1.Gauss Scaled Elimination. \n\t2.Iterative Gauss-Siedel\n";
		int s;
		cin >> s;
		cout << "============================" << endl;
		cout << "2D Ploynomial Regression" << endl;
		cout << "============================" << endl;

		cout << "///////////////////////////////////////" << endl;
		cout << "\nUsing \"bedrooms\" and \"bathrooms\"" << endl;
		int out = regress_2d(0, 1, s);
		cout << "\nUsing \"bedrooms\" and \"stories\"" << endl;
		out = regress_2d(0, 2, s);
		cout << "\nUsing \"bedrooms\" and \"lotsize\"" << endl;
		out = regress_2d(0, 3, s);
		cout << "\nUsing \"bathrooms\" and \"stories\"" << endl;
		out = regress_2d(1, 2, s);
		cout << "\nUsing \"bathrooms\" and \"lotsize\"" << endl;
		out = regress_2d(1, 3, s);
		cout << "\nUsing \"stories\" and \"lotsize\"" << endl;
		out = regress_2d(2, 3, s);


	}
	else if (number == 3) {
		cout << "Interpolation" << endl;

		cout << "\n===========================" << endl;
		cout << "\n\nUsing First Dataset:" << endl;
		cout << "===========================" << endl;

		interpolate("3_dataset_1.csv");

		cout << "===========================" << endl;
		cout << "\n\nUsing Second Dataset:" << endl;
		cout << "===========================" << endl;
		
		interpolate("3_dataset_2.csv");

	}
	else {
		cout << "Wrong Choice !" << endl;
	}
	int ss;
	cin >> ss;
	return 0;


}

