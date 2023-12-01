#pragma once
#include "serialisation.h"
#include <SFML/Graphics.hpp>
static int reverseInt(int i)
{
    unsigned char c1, c2, c3, c4;

    c1 = i & 255;
    c2 = (i >> 8) & 255;
    c3 = (i >> 16) & 255;
    c4 = (i >> 24) & 255;

    return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
}
std::vector<std::vector<float>> readTrainingImages(const std::string& filename, int debug)
{

    std::ifstream file(filename, std::ios::binary);

    if (file.is_open())
    {
        int32_t magicNumber = 0, numberOfImages = 0, rows = 0, cols = 0;

        file.read((char*)&magicNumber, sizeof(magicNumber));
        magicNumber = reverseInt(magicNumber);

        file.read((char*)&numberOfImages, sizeof(numberOfImages));
        numberOfImages = reverseInt(numberOfImages);
        if (debug) numberOfImages = debug;

        file.read((char*)&rows, sizeof(rows));
        rows = reverseInt(rows);

        file.read((char*)&cols, sizeof(cols));
        cols = reverseInt(cols);

        std::size_t numberOfImagesForIndex = numberOfImages, rowsForIndex = rows, colsForIndex = cols;
        //create a vector of size numberOfimages x (rows x cols) to store images in
        std::vector<std::vector<float>> images(numberOfImagesForIndex, std::vector<float>(rowsForIndex * colsForIndex));

        for (std::size_t i = 0; i < numberOfImagesForIndex; ++i)
        {

            for (std::size_t j = 0; j < rowsForIndex * colsForIndex; ++j)
            {
                unsigned char pixel;
                file.read((char*)&pixel, sizeof(unsigned char));
                images[i][j] = static_cast<float>(pixel) / 255.0f;
            }
        }

        std::cout << "Successfully loaded " << numberOfImages << " Images From " << filename << std::endl;
        file.close();

        return images;
    }
    else
    {
        std::cout << "Cannot open file: " << filename << std::endl;
        return std::vector<std::vector<float>>();
    }
}

std::vector<int> readTrainingLabels(const std::string filename)
{

    std::ifstream file(filename, std::ios::binary);

    if (file.is_open())
    {
        int32_t magicNumber = 0, numOfLabels = 0;

        file.read((char*)&magicNumber, sizeof(magicNumber));
        magicNumber = reverseInt(magicNumber);

        file.read((char*)&numOfLabels, sizeof(numOfLabels));
        numOfLabels = reverseInt(numOfLabels);

        std::size_t numOfLabelsForIndex = static_cast<std::size_t>(numOfLabels);


        std::vector<int> labels(numOfLabelsForIndex);

        for (std::size_t i = 0; i < numOfLabelsForIndex; ++i)
        {
            unsigned char label;
            file.read((char*)&label, sizeof(unsigned char));
            labels[i] = static_cast<int>(label);
        }

        file.close();
        std::cout << "Successfully loaded " << numOfLabelsForIndex << " Labels From " << filename << std::endl;
        return labels;
    }
    else
    {
        throw std::runtime_error("Cannot open file: " + filename);
    }
}
std::string convertTopologyToString(std::vector<std::size_t> topology) {
    std::string topologyString = "";
    for (std::size_t i = 0; i < topology.size(); ++i)
    {
        topologyString += std::to_string(topology[i]);
        if (i != topology.size() - 1) topologyString += "x";
    }
    return topologyString;
}
// Function to read weights and biases from a file
void readParamsFromFile(std::vector<std::size_t> topology, Network& network) {

    //convert topology to a string
    std::string topologyString = convertTopologyToString(topology);

    //add the topology to an initial filename

    std::string filename = "weightsAndBiases" + topologyString + ".bin";

    std::ifstream inFile(filename, std::ios::in | std::ios::binary);
    if (!inFile) {
        //if the file does not exist, create it
        throw std::runtime_error("Error opening File " + filename);
    }

    for (auto& biasLayer : network.biases) {
        inFile.read(reinterpret_cast<char*>(&biasLayer.data[0]), biasLayer.size() * sizeof(float));
    }
    for (auto& weightLayer : network.weights) {
        inFile.read(reinterpret_cast<char*>(&weightLayer.data[0]), weightLayer.size() * sizeof(float));
    }

    inFile.close();
}

void writeParamsToFile(const std::vector<std::size_t> topology, Network& network) {

    //convert topology to a string
    std::string topologyString = convertTopologyToString(topology);

    //add the topology to an initial filename
    std::string filename = "weightsAndBiases" + topologyString + ".bin";

    std::ofstream outFile(filename, std::ios::out | std::ios::binary);
    if (!outFile) {
        throw std::runtime_error("Error opening File " + filename);

    }

    //iterate through all the weights and biases and write them to the file
    for (auto& biasLayer : network.biases) {
        outFile.write(reinterpret_cast<const char*>( & biasLayer.data[0]), biasLayer.size() * sizeof(float));
    }
    for (auto& weightLayer : network.weights) {
        outFile.write(reinterpret_cast<const char*>(&weightLayer.data[0]), weightLayer.size() * sizeof(float));
    }

    outFile.close();

}
