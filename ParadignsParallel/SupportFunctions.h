#pragma once
#include<SFML/Graphics.hpp>
#define CLOCK chrono::steady_clock::now
#define CLOCK_TO_MS chrono::duration_cast<std::chrono::nanoseconds>
#define SHORT_IN_MM256 16
#define CHAR_IN_MM256 32

using namespace sf;

void drawImage(Image& image, RenderWindow& window, Vector2f pos)
{
	Texture texture;
	texture.loadFromImage(image);
	Sprite sprite;
	sprite.setTexture(texture);
	sprite.setPosition(pos);
	window.draw(sprite);
}

Image imageFromBytesRGB(unsigned char* pixels, unsigned int width, unsigned int height)
{
	int size = width * height * 4;
	unsigned char* newPixels = new unsigned char[size];
	for (int i = 0, j = 0; i < size; i++, j++)
	{
		if ((i + 1) % 4 == 0 && i != 0)
		{
			newPixels[i] = 255;
			j--;
			continue;
		}
		newPixels[i] = pixels[j];
	}
	Image newImage;
	newImage.create(width, height, newPixels);
	return newImage;
}


Image imageFromBytesRGBA(unsigned char* pixels, unsigned int width, unsigned int height)
{
	Image newImage;
	newImage.create(width, height, pixels);
	return newImage;
}

void bytesFromImageRGB(Image image, unsigned char*& pixels, int& size, unsigned int& width, unsigned int& height)
{
	const unsigned char* imagePixels = image.getPixelsPtr();
	width = image.getSize().x, height = image.getSize().y;
	size = width * height * 3;
	pixels = new unsigned char[size];
	for (int i = 0, j = 0; i < size; i++, j++)
	{
		if ((j + 1) % 4 == 0 && j != 0)
			j++;

		pixels[i] = imagePixels[j];
	}
}

void bytesFromImageRGBA(Image image, unsigned char*& pixels, int& size, unsigned int& width, unsigned int& height)
{
	const unsigned char* imagePixels = image.getPixelsPtr();
	width = image.getSize().x, height = image.getSize().y;
	size = width * height * 4;
	pixels = new unsigned char[size];
	for (int i = 0; i < size; i++) pixels[i] = imagePixels[i];
}

double gaussianFunction(int x, int y, double mean, double sigma)
{
	return exp(-0.5 * (pow((x - mean) / sigma, 2) + pow((y - mean) / sigma, 2))) / (2 * M_PI * sigma * sigma);
}

void buildCore(float*& core, int coreSize, float sigma)
{
	float mean = coreSize / 2;
	float coreSum = 0.0;
	for (int x = 0; x < coreSize; x++)
	{
		for (int y = 0; y < coreSize; y++)
		{
			core[x * coreSize + y] = gaussianFunction(x, y, mean, sigma);
			coreSum += core[x * coreSize + y];
		}
	}
	for (int i = 0; i < coreSize * coreSize; i++)
	{
		core[i] /= coreSum;
	}
}
//
//void printReg(__m256i& reg, string name)
//{
//	cout << name + " "; 
//	for (int i = 0; i < SHORT_IN_MM256; i++) 
//		cout << ((short*)&reg)[i] << " "; 
//	cout << endl;
//
//}

