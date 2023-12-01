#pragma once 
#include <SFML/Graphics.hpp>
#include "Network.h"
#include <vector>
#include <algorithm>
static bool onScreen(sf::Vector2i mousePos);
static sf::Vector2i getIndex(sf::Vector2i mousePos);
static bool validIndex(std::size_t posistion);
static std::size_t getRowMajorIndex(sf::Vector2i posistion);
static bool compareSecond(const std::pair<std::size_t, float>& a, const std::pair<std::size_t, float>& b);
static void setScreenAndInput(std::vector < std::vector<sf::RectangleShape>>& grid, std::vector<float>& inputToNetwork, sf::Vector2i mousePos, bool removing);
void drawAndPredict(Network& network);