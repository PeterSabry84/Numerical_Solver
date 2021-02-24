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
//Normalizing features matrix
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
//Normalizing output vector
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
//Denormalizing predicted output >> used in reporting the prediction
double denorm_y(const double y_norm, const double y_min, const double y_max) {
	double y_denorm = y_norm * (y_max - y_min) + y_min;
	return y_denorm;
}

//Calculate mean-square error
double calc_mse(const double y[], const double y_pred[], const int n) {
	double mse = 0;
	for (int i = 0; i < n; i++)
		mse = mse + pow((y[i] - y_pred[i]),2) /n;
	return mse;
}

// Starts single variable regression given:
//		filename:	Input CSV file name
//		m:			Polynomial Degree desired
//		s:			Solver selection (1. Gauss Elimination, 2. Gauss-Seidel)
void regress_line(string filename, int m, int s) {
	int i, n;
	
	// Reading training input data file, construct x[], y[] 
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


	//Initialize and fit the regressor
	univar_regressor polyRegress = univar_regressor(x, y, m, s);
	Matrix coeff = polyRegress.fit(x, y, n, m, s);

	//Retrieve the ill_conditioned flag and number of gauss-seidel iterations if any
	bool valid_sol = polyRegress.is_valid_solution();
	int num_iterations = polyRegress.seidel_iterations;

	if (valid_sol)
		cout << "System has valid Solution" << endl;
	else
		cout << "System is ill conditioned" << endl;

	//Print the coefficients to the console
	cout << "\nThe values of the solution coefficients are:\n";
	for (int idx = 0; idx <= m; idx++)
		cout << "x^" << idx << "=" << coeff.at(idx, 0) << endl;            // Print the values of x^0,x^1,x^2,x^3,....    
	
	//Print the fitted equation
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

// Starts 2-D variable regression for the Housing datset,
// Housing dataset contains 4 features, taken two of them at a time
//   regression is done 6 times, one for each possible combination of two features
//	Input:
//		x1			The column index in the file for the first feature
//		x2			The column index in the file for the second feature
//		s:			Solver selection (1. Gauss Elimination, 2. Gauss-Seidel)
int regress_2d(const int x1, const int x2, const int s) {  
	int i, j, n, m;
	m = 2;    // two dimensional input x: [x1, x2]
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

	//Populate the input matrix
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

	//Initialize and fit the 2D regressor
	multivar_regressor polyRegress2D = multivar_regressor(x_norm, y_norm, n, m, s);
	Matrix coeff = polyRegress2D.fit(x_norm, y_norm, n, m, s);

	//Print the coefficients
	cout << "\nThe values of the solution coefficients are:\n";
	for (int idx = 0; idx <= m; idx++)
		cout << "x_" << idx << "=" << coeff.at(idx, 0) << endl;            // Print the values of x^0,x^1,x^2,x^3,....    
	//Print the equation
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

	//De-normalizing prediction
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

//Starts the interpolation given an input file name
//It calls both newton and spline interpolators
void interpolate(string filename) {


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
	//Newton interpolation
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

	//Print the coefficients
	cout << s_coeff;
	cout << endl;

	//Output to file
	string s_coeff_filename = "spline_coeff_" + filename;
	ofstream s_outfile;
	outfile.open(coeff_filename);
	outfile << s_coeff;
 
	outfile.close();
	
	//doubling the points x and y.
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

	//Doubling the points once more
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


/// The main application function, it calls all the above functions. It asks for the following choices
/// 0. Apply equation solvers to the test case in the assignment
/// 1. Do single variable regression
///			1. Using Gauss Elimination
///			2. Usnig Gauss-Seidel
/// 2. Do 2D regression on Housing dataset
///			1. Using Gauss Elimination
///			2. Usnig Gauss-Seidel
/// 3. Do interpolation
/// 
int main()
{
	int number = 0;
	cout << "Chose what do you want to do: \n \t0: Gauss-Seidel test. \n \t1: Single variable Polynomial Regression. \n\t2: 2D Ploynomial Regression. \n\t3: Interpolation. \n";
	cin >> number;

	//First Choice, Test the equation solvers
	if (number == 0)
	{
		double test_case[12] = { 2, 1, -1, 0, 1, 4, 3, 14, -1, 2, 7, 30};
		Matrix AB(3, 4, test_case);
		Linear_system ls(3, 4, AB);
		cout << "\nUsing Gauss elimination:\n" << ls.solve() << endl;
		double initials[3] = { 0, 0, 0 };
		Matrix AB1(3, 4, test_case);
		Linear_system ls1(3, 4, AB1);
		int n_ = 0;
		cout << "\nUsing Seidel elimination with Gauss-Seidel: \n" << ls1.solve_until(initials, n_, 0.00000001) << endl;
		cout << "\nNumber of iterations: " << n_ << endl;
	}

	// Second Choice: Linear Regression
	else if (number == 1)
	{
		// Ask for the desired polynomial degree to fit
		//   the higher the order, the better the fitting, more calculation
		int m;
		cout << "Linear Regression" << endl;
		cout << "\nWhat degree of Polynomial do you want to use for the fit?\n";
		cin >> m;

		//Ask for the desired equation solver
		cout << "\n Chose to Sove using: \n\t1.Gauss Scaled Elimination. \n\t2.Iterative Gauss-Siedel\n";
		int s;
		cin >> s;

		cout << "===========================" << endl;
			cout << "\n\nUsing First Dataset:" << endl;
			cout << "===========================" << endl;
			
			//Call the regressor for the first dataset
			regress_line("2_a_dataset_1.csv", m, s);

			cout << "\n===========================" << endl;
			cout << "\n\nUsing Second Dataset:" << endl;
			cout << "===========================" << endl;

			//Call the regressor for the second dataset
			regress_line("2_a_dataset_2.csv", m, s);
	

	}
	// Third Choice: 2D regression on Housing dataset
	else if (number == 2) {

		//Ask to select the equation solver
		cout << "\n Chose to Sove using: \n\t1.Gauss Scaled Elimination. \n\t2.Iterative Gauss-Siedel\n";
		int s;
		cin >> s;
		cout << "============================" << endl;
		cout << "2D Ploynomial Regression" << endl;
		cout << "============================" << endl;

		//Call the 2D regressor for each combination of 2 features of the 4 available features
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
	// Forth Coice: Do the interpolation
	else if (number == 3) {
		cout << "Interpolation" << endl;

		cout << "\n===========================" << endl;
		cout << "\n\nUsing First Dataset:" << endl;
		cout << "===========================" << endl;

		//Call the interpolator for the first dataset
		interpolate("3_dataset_1.csv");

		cout << "===========================" << endl;
		cout << "\n\nUsing Second Dataset:" << endl;
		cout << "===========================" << endl;
		
		//Call the interpolator for the second dataset
		interpolate("3_dataset_2.csv");

	}
	else {
		cout << "Wrong Choice !" << endl;
	}
	int ss;
	cin >> ss;
	return 0;

}

