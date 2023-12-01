#pragma once
#include "Matrix.h"
#include "activationAndInitFunctions.h"
using std::to_string;

struct backpropStorage {
	std::vector<Matrix> errors;
	std::vector<Matrix> weightGradients;
	void init(std::vector<std::size_t> topology) {
		for (std::size_t layer = 1; layer < topology.size(); ++layer)
		{
			errors.push_back(Matrix(topology[layer], 1));
		}
		for (std::size_t layer = 0; layer < topology.size() - 1; ++layer)
		{
			weightGradients.push_back(Matrix(topology[layer + 1], topology[layer]));
		}
	}
};
class Network 
{

public:
		

	std::vector<std::size_t > topology;
	std::vector<Matrix> layers;
	std::vector<Matrix> weights;
	std::vector<Matrix> biases;
	std::vector<Matrix> intermediates;

	backpropStorage batchDerivatives;
	backpropStorage currentDerivatives;

	std::vector<funcAndDerivative> activationFunctions;
	costAndDerivative costFunction;



	std::size_t numberOfLayers;

	Network(std::vector<std::size_t> topology, std::vector<funcAndDerivative> activationFunctionsInp,
		costAndDerivative costFunctionInp,  void (*weightsInitialisation)(Matrix&))
		: numberOfLayers(topology.size()), topology(topology), activationFunctions(activationFunctionsInp), costFunction(costFunctionInp)
	{

			numberOfLayers = topology.size();
			layers.reserve(numberOfLayers);
			weights.reserve(numberOfLayers - 1);
			biases.reserve(numberOfLayers);
			intermediates.reserve(numberOfLayers);

			this->topology = topology;

			batchDerivatives.init(topology);
			currentDerivatives.init(topology);

			

			//add input layer
			layers.push_back(Matrix(topology.front(), 1, "Layer 0 "));
			
			//add the rest of the layers, and their respective, biases, intermediates and error matrices
			for (std::size_t layer = 1; layer < topology.size(); ++layer)
			{
				std::string current_name = "Layer " + to_string(layer);
				layers.push_back(Matrix(topology[layer], 1, current_name));
				current_name = "Biases for layer " + to_string(layer);
				biases.push_back(Matrix(topology[layer], 1, current_name));
				current_name = "Error for layer " + to_string(layer);
				current_name = "Intermediate values for layer " + to_string(layer);
				intermediates.push_back(Matrix(topology[layer], 1, current_name));
			}
			//add the weights, and their respective partial derivative matrices
			for (std::size_t layer = 0; layer < topology.size() - 1; ++layer)
			{
				std::string current_name = "Weights connecting layer " + to_string(layer) + " to layer " + to_string(layer + 1);
				weights.push_back(Matrix(topology[layer + 1], topology[layer], current_name));
				weightsInitialisation(weights[layer]);
				current_name = "Partial derivatives for Weights connecting layer " + to_string(layer) + " to layer " + to_string(layer + 1);
			}

		}
		void feedForward(std::vector<float> input) 
		{	
			layers.front().data = input; //copy input to first layer
			for (std::size_t layerIndex = 0; layerIndex < topology.size() - 1; ++layerIndex)
			{
				intermediates[layerIndex] = (weights[layerIndex] * layers[layerIndex]) + biases[layerIndex]; //calculate intermediate values
				layers[layerIndex + 1] = activationFunctions[layerIndex].activation(intermediates[layerIndex]); //apply activation function to intermediate values to get layer values			
			}
		}
		void backpropagate(int label) //this overides the batch derivatives, so use backpropagate sum for 
		{
			//calculate errors for output layer
			Matrix correct(topology[numberOfLayers - 1], 1, "correct matrix");
			correct(label, 0) = 1;
			
			//^ represents element wise multiplication
			
			currentDerivatives.errors.back() = (layers.back() - correct) ^ (activationFunctions.back().derivative(intermediates.back())); //calculate error for output layer

			currentDerivatives.weightGradients.back() = currentDerivatives.errors.back() * layers[numberOfLayers - 2].transposition(); //calculate partial derivatives for weights connecting output layer to previous layer

			for (std::size_t current_layer = numberOfLayers - 2; current_layer != 0; --current_layer)
			{
				currentDerivatives.errors[current_layer] = (weights[current_layer].transposition() * currentDerivatives.errors[current_layer + 1]) ^ activationFunctions[current_layer].derivative(intermediates[current_layer - 1]);
				//errors[current_layer].print("error for layer " + to_string(current_layer) + ": ");
				currentDerivatives.weightGradients[current_layer - 1] = currentDerivatives.errors[current_layer] * layers[current_layer - 1].transposition();
			}

			for (std::size_t error = 0; error < currentDerivatives.errors.size(); error++)
			{
				batchDerivatives.errors[error] += currentDerivatives.errors[error];
			}
			for (std::size_t derivative = 0; derivative < currentDerivatives.weightGradients.size(); derivative++)
			{
				batchDerivatives.weightGradients[derivative] += currentDerivatives.weightGradients[derivative];
			}
	
		}

		void adjust(float learningRate) 
		{

			for (std::size_t layer = 0; layer < topology.size() - 1; ++layer)
			{
				//biases[layer].print("biases before adjustment: ");
				weights[layer] -= (batchDerivatives.weightGradients[layer] * learningRate);
				batchDerivatives.weightGradients[layer].setZero();
				biases[layer] -= (batchDerivatives.errors[layer + 1] * learningRate);
				batchDerivatives.errors[layer].setZero();
				//(errors[layer + 1] * learningRate).print("error times learning rate: ");
				//biases[layer].print("biases after adjustment: ");
			}
		}
		void l2Regularise(float lambda)
		{
			for (std::size_t layer = 0; layer < topology.size() - 1; ++layer)
			{
				weights[layer] = weights[layer] - (weights[layer] * lambda);
			}
		}
		

};