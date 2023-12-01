#pragma once
#include <vector>
#include <iostream>
#include <string>
#include <random>
#include <chrono>
struct Matrix
{
	std::vector<float> data;
	std::size_t rows, cols;
	std::string name;

	//matrices are initialised to zero by default
	Matrix(std::size_t rows, std::size_t cols, std::string name = "Unnamed matrix") : data(rows* cols), rows(rows), cols(cols), name(name) {
		for(auto& x: data)
		{
			x = 0.0f;
		}
	}

	//overloaded () operator to access elements of the matrix as if it were a 2d array
	float& operator()(std::size_t row, std::size_t col){
		return data[(row * cols) + col];	
	}
	const float& operator()(std::size_t row, std::size_t col) const{
		return data[(row * cols) + col];
	}

	//print rows, cols, and name of matrix
	void printDebugInfo() const{
		std::cout << "Name: " << name << " Rows: " << rows << " Cols: " << cols << "\n";
	}

	//print the dimensions of the matrix
	void  printDimensions() const
	{
		std::cout << "Matrix (" << name << ") is " << rows << "x" << cols << "\n";
	}

	//print all values in the matrix
	void print(std::string prefix = " ") const {

		std::cout << prefix << " ";
		if (cols > 1)
		{
			for (std::size_t row = 0; row < rows; ++row) {
				for (std::size_t col = 0; col < cols; ++col) {
					std::cout << (*this)(row, col) << " ";
				}
				std::cout << "\n";
			}
		}
		else //print column vectors as row vectors for ease of reading
		{
			for (std::size_t row = 0; row < rows; ++row) {
				std::cout << (*this)(row, 0) << " ";
			}
			std::cout << "\n";
		}

	}

	//matrix addition with another matrix
	Matrix operator+(const Matrix& other) const{
		if (rows != other.rows || cols != other.cols) throw std::runtime_error("Matrix dimensions do not match for addition: " + name + " (" + std::to_string(rows) + "x" + std::to_string(cols) + ") and " + other.name + " (" + std::to_string(other.rows) + "x" + std::to_string(other.cols) + ")");
		Matrix result(rows, cols, "result of adding (" + name + ") and (" + other.name + ")");
		for (std::size_t row = 0; row < rows; ++row)
		{
			for (std::size_t col = 0; col < cols; ++col)
			{
				result(row, col) = data[row * cols + col] + other(row, col);
			}
		}
		return result;
	}

	void operator +=(const Matrix & other) {
		if (rows != other.rows || cols != other.cols) throw std::runtime_error("Matrix dimensions do not match for addition: " + name + " (" + std::to_string(rows) + "x" + std::to_string(cols) + ") and " + other.name + " (" + std::to_string(other.rows) + "x" + std::to_string(other.cols) + ")");
		for (std::size_t row = 0; row < rows; ++row)
		{
			for (std::size_t col = 0; col < cols; ++col)
			{
				(*this)(row,col) += other(row, col);
			}
		}

	}
	
	//matrix subtraction with another matrix
	Matrix operator-(const Matrix& other) const{
		if (rows != other.rows || cols != other.cols) throw std::runtime_error("Matrix dimensions do not match for subtraction: " + name + " (" + std::to_string(rows) + "x" + std::to_string(cols) + ") and " + other.name + " (" + std::to_string(other.rows) + "x" + std::to_string(other.cols) + ")");
		Matrix result(rows, cols, "Result of subtracting (" + other.name + ") from (" + name + ")");
		for (std::size_t row = 0; row < rows; ++row)
		{
			for (std::size_t col = 0; col < cols; ++col)
			{
				result(row, col) = (*this)(row, col) - other(row, col);
			}
		}
		return result;
	}

	void operator -=(const Matrix& other) {
		if (rows != other.rows || cols != other.cols) throw std::runtime_error("Matrix dimensions do not match for addition: " + name + " (" + std::to_string(rows) + "x" + std::to_string(cols) + ") and " + other.name + " (" + std::to_string(other.rows) + "x" + std::to_string(other.cols) + ")");
		for (std::size_t row = 0; row < rows; ++row)
		{
			for (std::size_t col = 0; col < cols; ++col)
			{
				(*this)(row, col) -= other(row, col);
			}
		}

	}

	//matrix multiplication with another matrix 
	Matrix operator*(const Matrix& other) const{
		if (cols != other.rows) throw std::runtime_error("Matrix dimensions do not match for multiplication: " + name + " (" + std::to_string(rows) + "x" + std::to_string(cols) + ") and " + other.name + " (" + std::to_string(other.rows) + "x" + std::to_string(other.cols) + ")");
		Matrix result(rows, other.cols, "result of multiplying (" + name + ") and (" + other.name + ")");
		for (std::size_t row = 0; row < rows; ++row)
		{
			for (std::size_t col = 0; col < other.cols; ++col)
			{
				float sum = 0.0;
				for (std::size_t k = 0; k < cols; ++k)
				{
					sum += (*this)(row, k) * other(k, col);
				}
				result(row, col) = sum;
			}
		}
		return result;
	}

	//tiled matrix multiply, as our data is stored in row major order this is faster than the normal matrix multiplication (this is slower?)
	Matrix tiledMatrixMultiply(const Matrix& other, std::size_t tileSize) {
		
		std::size_t K = other.cols;
		if (cols != other.rows) throw std::runtime_error("Matrix dimensions do not match");

		Matrix C(rows, K, "Result of multiplying (" + name + ") and (" + other.name + ")");
		for (std::size_t i = 0; i < rows; i += tileSize) {
			for (std::size_t j = 0; j < K; j += tileSize) {
				for (std::size_t k = 0; k < cols; k += tileSize) {
					for (std::size_t ii = i; ii < std::min(i + tileSize, rows); ++ii) {
						for (std::size_t jj = j; jj < std::min(j + tileSize, K); ++jj) {
							float sum = 0.0;
							for (std::size_t kk = k; kk < std::min(k + tileSize, cols); ++kk) {
								sum += (*this)(ii, kk) * other(kk, jj);
							}
							C(ii, jj) += sum;
						}
					}
				}
			}
		}

		return C;
	}

	//matrix multiplication with a scalar (overload)
	Matrix operator*(const float scalar) const{
		Matrix result(rows, cols, "result of multiplying (" + name + ") and (" + std::to_string(scalar) + ")");
		for (std::size_t row = 0; row < rows; ++row)
		{
			for (std::size_t col = 0; col < cols; ++col)
			{
				result(row, col) = data[row * cols + col] * scalar;
			}
		}
		return result;
	}

	Matrix operator *=(const float scalar) {
		for (std::size_t row = 0; row < rows; ++row)
		{
			for (std::size_t col = 0; col < cols; ++col)
			{
				(*this)(row,col) *= scalar;
			}
		}
	}

	//doesnt transpose this matrix, returns a transposition of this matrix
	Matrix transposition() const {
		Matrix transposition(cols, rows, "transposition of (" + name + ")");
		for (std::size_t row = 0; row < rows; ++row)
		{
			for (std::size_t col = 0; col < cols; ++col)
			{
				transposition(col, row) = (*this)(row, col);
			}
		}
		return transposition;
	}

	//just so that we dont accidentally copy the name, or change the dimensions of a matrix
	Matrix operator=(const Matrix& other){
		if (other.rows != rows || other.cols != cols) throw std::runtime_error("Cannot set matrices of these dimensions equal to one another(" + name + " and " + other.name +")");
		data = other.data;
		return (*this);
	}

	//hadamard product
	Matrix operator^(const Matrix& other) const{
		//create a result matrix with the same dimensions as a and b, whose name indicates that it is the hadamard product of a and b
		if (rows != other.rows || cols != other.cols) throw std::runtime_error("Matrix dimensions do not match for multiplication: " + name + " (" + std::to_string(rows) + "x" + std::to_string(cols) + ") and " + other.name + " (" + std::to_string(other.rows) + "x" + std::to_string(other.cols) + ")");
		Matrix result(rows, other.cols, "hadamard product of (" + name + ") and (" + other.name + ")");
		for (std::size_t row = 0; row < rows; ++row)
		{
			for (std::size_t col = 0; col < cols; ++col)
			{
				result(row, col) = (*this)(row, col) * other(row, col);
			}
		}
		return result;
	}

	//define .begin() to point to the first element of the matrix
	std::vector<float>::iterator begin(){
			return data.begin();
	}

	//and .end() to point to the last element of the matrix
	std::vector<float>::iterator end(){
				return data.end();
	}

	//define .size() to return the number of elements in the matrix
	std::size_t size() const {
		return data.size();
	}

	void setZero() {
		for (auto& elem : data) {
			elem = 0.0f;
		}
	}
	
};








