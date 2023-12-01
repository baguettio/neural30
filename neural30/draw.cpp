#pragma once 
#include "draw.h"

constexpr std::size_t squareSize = 32, rows = 28, drawingWindow = rows * rows;
constexpr std::size_t predictionsSize = 180;
constexpr std::size_t screenWidth = drawingWindow + predictionsSize, screenHeight = (drawingWindow);
static bool onScreen(sf::Vector2i mousePos) {
	return !((mousePos.x > screenWidth) || (mousePos.y > screenHeight));
}
static sf::Vector2i getIndex(sf::Vector2i mousePos) {
	return sf::Vector2i(round(mousePos.x / squareSize), round(mousePos.y / squareSize));
}
static bool validIndex(std::size_t posistion) {
	return !((posistion < 0) || (posistion > rows));
}
static std::size_t getRowMajorIndex(sf::Vector2i posistion) {
	return ((posistion.y * rows) + posistion.x);
}
static bool compareSecond(const std::pair<std::size_t, float>& a, const std::pair<std::size_t, float>& b) {
	return a.second > b.second;
}
static void setScreenAndInput(std::vector < std::vector<sf::RectangleShape>>& grid, std::vector<float>& inputToNetwork, sf::Vector2i mousePos, bool removing)
{

	sf::Vector2i index = getIndex(mousePos); //get the index of the square the mouse was over when it was clicked
	constexpr float greyVal = 0.7, greyscaleVal = greyVal * 255;
	if (!removing)
	{
		grid[index.y][index.x].setFillColor(sf::Color::Black); //set the square clicked to black
		inputToNetwork[getRowMajorIndex(index)] = 1; //set value of our input to the neural network to the same
	}
	else
	{
		grid[index.y][index.x].setFillColor(sf::Color::White); //if removing, erase 
		inputToNetwork[getRowMajorIndex(index)] = 0;
	}

}
void drawAndPredict(Network& network)
{

	std::vector<std::vector<sf::RectangleShape>> grid;
	std::vector<float> inputToNetwork(rows * rows);

	std::vector<sf::Text> predicitons(10);

	// Populate the grid with sf::RectangleShape objects
	for (std::size_t i = 0; i < rows; ++i)
	{
		std::vector<sf::RectangleShape> row;
		for (std::size_t x = 0; x < rows; ++x)
		{
			sf::RectangleShape currentSquare(sf::Vector2f(squareSize, squareSize));
			currentSquare.setPosition((x * squareSize), (i * squareSize));
			//squares are white by default so we dont need to change anything
			row.push_back(currentSquare);
		}
		grid.push_back(row);
	}

	sf::RenderWindow window(sf::VideoMode(1920, 1080), "My window");

	// run the program as long as the window is open
	while (window.isOpen())
	{

		//we handle mouse clicks outside the events window in order to support click and drag
		bool updateScreen = false;
		if (sf::Mouse::isButtonPressed(sf::Mouse::Left))
		{
			sf::Vector2i mousePos = sf::Mouse::getPosition(window); //get posistion of mouse relative to window
			if (onScreen(mousePos)) //if the mouse is on screen
			{
				setScreenAndInput(grid, inputToNetwork, mousePos, false);
				updateScreen = true;
			}
		}
		if (sf::Mouse::isButtonPressed(sf::Mouse::Right)) //exactly the same expcet right mouse removes stuff
		{
			sf::Vector2i mousePos = sf::Mouse::getPosition(window);
			std::cout << mousePos.x << " " << mousePos.y << std::endl;
			if (onScreen(mousePos)) //if the mouse is on screen
			{
				setScreenAndInput(grid, inputToNetwork, mousePos, true);
				updateScreen = true;
			}
		}

		// check all the window's events that were triggered since the last iteration of the loop
		sf::Event event;
		while (window.pollEvent(event))
		{
			// "close requested" event: we close the window
			if (event.type == sf::Event::Closed)window.close();

		}

		if (event.key.scancode == sf::Keyboard::Scan::Escape)
		{
			for (auto& x : grid)
			{
				for (auto& y : x)
				{
					y.setFillColor(sf::Color::White);
				}
			}

			std::fill(inputToNetwork.begin(), inputToNetwork.end(), 0.0f);

			updateScreen = true;
		}

		if (updateScreen)
		{
			network.feedForward(inputToNetwork);

			//create a vector of pairs binding each digit to its respective output
			std::vector<std::pair<size_t, float>> digitAndOutput;
			std::size_t digit = 0;
			for (const auto& output : network.layers.back())
			{
				digitAndOutput.push_back(std::make_pair(digit, output));
				++digit;
			}

			//sort this vector according to the output (second element)
			std::sort(digitAndOutput.begin(), digitAndOutput.end(), compareSecond);
			std::vector<sf::Text> predictons;
			std::size_t yPos = 10;

			sf::Font font;

			// Load it from a file
			if (!font.loadFromFile("arial.ttf"))
			{
				throw std::runtime_error("ae");
			}

			for (const auto& x : digitAndOutput)
			{
				sf::Text text;

				text.setFont(font);

				// set the string to display
				text.setString(std::to_string(x.first) + ": " + std::to_string(x.second));

				// set the character size
				text.setCharacterSize(24); // in pixels, not points!

				// set the color
				text.setFillColor(sf::Color::Red);

				// set the text style
				text.setStyle(sf::Text::Bold | sf::Text::Underlined);

				text.setPosition(1000, yPos);

				text.setScale(3, 3);

				yPos += 100;

				predictons.push_back(text);
			}

			window.clear();



			for (auto& prediction : predictons)
			{
				window.draw(prediction);
			}

			for (auto& row : grid)
			{
				for (auto& square : row)
				{
					window.draw(square);
				}

			}

			window.display();

		}
	}

}