#pragma once
#include <functional>

constexpr float max = 0.001, min = -0;
inline void randomise(Matrix& vec) {
	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_real_distribution<float> distribution(min, max);

	for (auto& element : vec) {
		element = distribution(gen);
	}
}
inline void setZero(Matrix& matrix)
{
	for (auto& x : matrix)
	{
		x = 0.0f;
	}
}
struct funcAndDerivative {
	Matrix(*activation)(const Matrix&) = nullptr;
	Matrix(*derivative)(const Matrix&) = nullptr;

	funcAndDerivative(Matrix (*activation)(const Matrix&), Matrix(*derivative)(const Matrix&))
		: activation(activation), derivative(derivative) {}
};
struct costAndDerivative {
	Matrix(*activation)(const Matrix&, const Matrix&) = nullptr;
	Matrix(*derivative)(const Matrix&, const Matrix&) = nullptr;

	costAndDerivative(Matrix(*activationFunc)(const Matrix&, const Matrix&), Matrix(*derivativeFunc)(const Matrix&, const Matrix&))
		: activation(activationFunc), derivative(derivativeFunc) {}
};

inline Matrix sigmoidImplementation(const Matrix& matrix)
{
	Matrix sm(matrix.rows, matrix.cols, "sigmoid of (" + matrix.name + ")");
	for (std::size_t row = 0; row < matrix.rows; ++row)
	{
		for (std::size_t col = 0; col < matrix.cols; ++col)
		{
			sm(row, col) = 1 / (1 + exp(-matrix(row, col)));
		}
	}
	return sm;
}
inline Matrix sigmoidDerivativeImplementation(const Matrix& matrix)
{
	Matrix cm(matrix.rows, matrix.cols, "sigmoid derivative of (" + matrix.name + ")");
	for (std::size_t row = 0; row < matrix.rows; ++row)
	{
		for (std::size_t col = 0; col < matrix.cols; ++col)
		{
			cm(row, col) = matrix(row, col) * (1 - matrix(row, col));
		}
	}
	return cm;
}
inline Matrix ReLUImplementation(const Matrix& matrix)
{
	Matrix cm(matrix.rows, matrix.cols, "ReLU of (" + matrix.name + ")");
	for (std::size_t row = 0; row < matrix.rows; ++row)
	{
		for (std::size_t col = 0; col < matrix.cols; ++col)
		{
			cm(row, col) = (matrix(row, col) > 0) ? matrix(row, col) : 0;
		}
	}
	return cm;
}
inline Matrix ReLUDerivativeImplementation(const Matrix& matrix)
{
	Matrix cm(matrix.rows, matrix.cols, "ReLU derivative of (" + matrix.name + ")");
	for (std::size_t row = 0; row < matrix.rows; ++row)
	{
		for (std::size_t col = 0; col < matrix.cols; ++col)
		{
			cm(row, col) = matrix(row, col) > 0 ? 1 : 0;
		}
	}
	return cm;
}
inline Matrix softmaxImplementation(const Matrix& input) {
	Matrix result(input.rows, input.cols, "Softmax of (" + input.name + ")");

	// Compute softmax for each row
	for (std::size_t row = 0; row < input.rows; ++row) {
		// Find the maximum value in the row for numerical stability
		float maxVal = input(row, 0);
		for (std::size_t col = 1; col < input.cols; ++col) {
			maxVal = std::max(maxVal, input(row, col));
		}

		// Compute softmax for each element in the row
		float sumExp = 0.0f;
		for (std::size_t col = 0; col < input.cols; ++col) {
			float expVal = std::exp(input(row, col) - maxVal);
			result(row, col) = expVal;
			sumExp += expVal;
		}

		// Normalize the row
		for (std::size_t col = 0; col < input.cols; ++col) {
			result(row, col) /= sumExp;
		}
	}

	return result;
}
inline Matrix softmaxDerivativeImplementation(const Matrix& softmaxOutput) {
	Matrix result(softmaxOutput.rows, softmaxOutput.cols, "Softmax Derivative of (" + softmaxOutput.name + ")");

	for (std::size_t row = 0; row < softmaxOutput.rows; ++row) {
		for (std::size_t col = 0; col < softmaxOutput.cols; ++col) {
			float softmaxValue = softmaxOutput(row, col);
			result(row, col) = softmaxValue * (1.0f - softmaxValue);

			// Adjust the diagonal elements
			if (row == col) {
				result(row, col) -= softmaxValue;
			}
		}
	}

	return result;
}
inline Matrix mseImplementation(const Matrix& output, const Matrix& correct )
{
	Matrix result(output.rows, output.cols, "MSE applied to " + output.name);
	for (std::size_t x = 0; x < output.rows; x++)
	{
		for (std::size_t y = 0; y < output.cols; y++)
		{
			result(x, y) = pow(output(x, y) - correct(x, y), 2);
		}
	}
	return result;
}
inline Matrix meseDerivativeImplementation(const Matrix& output, const Matrix& correct)
{
	Matrix result(output.rows, output.cols, "MSE applied to " + output.name);
	for (std::size_t x = 0; x < output.rows; x++)
	{
		for (std::size_t y = 0; y < output.cols; y++)
		{
			result(x, y) = (output(x, y) - correct(x, y)) * 2;
		}
	}
	return result;
}
