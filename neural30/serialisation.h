#pragma once
#include "Matrix.h"
#include "Network.h"
#include <fstream>
#include <cmath>
//reverses the endianness of an integer
static int reverseInt(int i);
//reads training images from a file and returns it as a vector, if debug is set to non zero number, hen only that many images will be read
std::vector<std::vector<float>> readTrainingImages(const std::string& filename, int debug = 0);
//reads training labels from a file and returns it as a vector
std::vector<int> readTrainingLabels(const std::string filename);
//converts a topology vector to a string
std::string convertTopologyToString(std::vector<std::size_t> topology);
//reads the parameters of a network from a file, the filename is reated from the topology of the network
void readParamsFromFile(std::vector<std::size_t> topology, Network& network);
//writes the parameters of a network to a file, the filename is created from the topology of the network
void writeParamsToFile(const std::vector<std::size_t> topology, Network& network);
