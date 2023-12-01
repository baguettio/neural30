#include "Network.h"
#include "Matrix.h"
#include "activationAndInitFunctions.h"
#include "serialisation.h"
#include "draw.h"

//checks if network gets guess correct for every image in the testing set, returns number of correct guesses 
static void test(std::vector<std::vector<float>>& testImages, std::vector<int>& testLabels, Network& network)
{

    std::size_t numOfTestingImages = testImages.size();
    int correctC = 0;
    for (std::size_t current_image = 0; current_image < numOfTestingImages; ++current_image) //test the accuracy of the network on the test data
    {
        int label = testLabels[current_image];
        network.feedForward(testImages[current_image]);
        auto it = std::max_element(network.layers.back().begin(), network.layers.back().end());
        __int64 index = std::distance(network.layers.back().begin(), it);
        if (index == label) correctC++;

    }
    std::cout << "Correct: " << correctC << "/" << testImages.size() << std::endl;
}

int main()
{
    funcAndDerivative sigmoid(sigmoidImplementation, sigmoidDerivativeImplementation);
    funcAndDerivative ReLU(ReLUImplementation, ReLUDerivativeImplementation);
    funcAndDerivative softmax(softmaxImplementation, softmaxDerivativeImplementation);

    costAndDerivative MSE(mseImplementation, meseDerivativeImplementation);

    std::vector<std::size_t> topology = { {784, 15, 10} };
    Network network({ 784, 15, 10 }, { ReLU, ReLU, ReLU }, MSE, randomise);   //create a network with 784 input neurons, 256 hidden neurons, and 10 output neurons
    std::vector<std::vector<float>> images = readTrainingImages("train-images.idx3-ubyte");
    std::vector<int> labels = readTrainingLabels("train-labels.idx1-ubyte");

    std::vector<std::vector<float>> testImages = readTrainingImages("t10k-images.idx3-ubyte");
    std::vector<int> testLabels = readTrainingLabels("t10k-labels.idx1-ubyte");
    float learningRate = 0.01f;

    readParamsFromFile(topology, network);






    
    std::size_t epochs = 30, numberOfImages = images.size();
    test(testImages, testLabels, network);
    for (std::size_t epoch = 0; epoch < epochs; ++epoch)
    {
        std::cout << "Epoch " << epoch + 1 << "\n";
        for (std::size_t current_image = 0; current_image < numberOfImages; ++current_image)
        {
            network.feedForward(images[current_image]);
            network.backpropagate(labels[current_image]);
            network.adjust(learningRate);
        }
        //test(testImages, testLabels, network);
        learningRate *= 0.99f;

    }
    char ans = ' ';
    while (true)
    {
        std::cout << "write to file?" << std::endl;

        std::cin >> ans;
        if (ans == 'y'){
            writeParamsToFile(topology, network);
            break;
        }
        if (ans == 'n') break;
    }

    drawAndPredict(network);


    
	return 0;
	
	
}
