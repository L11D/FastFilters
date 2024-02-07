#define _CRT_SECURE_NO_WARNINGS
#define _USE_MATH_DEFINES
#include<iostream>
#include<SFML/Window.hpp>
#include<SFML/Graphics.hpp>
#include<vector>
#include<chrono>
#include<cmath>
#include<omp.h>
#include"SupportFunctions.h"
#include"OpenCLFuntions.h"
#include"IntrinsicsFuntions.h"
using namespace std;
using namespace sf;

#pragma optimize("", off)
Image gaussianBlur(Image image, int radius, float sigma, double& t)
{
	unsigned char* p;
	int size;
	unsigned int width, height;
	bytesFromImageRGB(image, p, size, width, height);

	int coreSize = 2 * radius + 1;
	float* core = new float[coreSize * coreSize];
	buildCore(core, coreSize, sigma);

	auto start = CLOCK();

	#pragma omp parallel for
	for (int x = 0; x < width; x++)
	{
		for (int y = 0; y < height; y++)
		{
			double r = 0, g = 0, b = 0;
			for (int i = -radius; i <= radius; i++)
			{
				int deltaX = 0;
				if (x + i < 0 || x + i >= width) deltaX = -i;
				else deltaX = i;

				for (int j = -radius; j <= radius; j++)
				{
					int deltaY = 0;
					if (y + j < 0 || y + j >= height) deltaY = -j;
					else deltaY = j;

					int index = 3 * ((y + deltaY) * width + x + deltaX);
					r += core[(i + radius) * coreSize + j + radius] * p[index];
					g += core[(i + radius) * coreSize + j + radius] * p[index + 1];
					b += core[(i + radius) * coreSize + j + radius] * p[index + 2];
				}
			}

			int index = 3 * (y * width + x);
			p[index] = r;
			p[index + 1] = g;
			p[index + 2] = b;
		}
	}
	double time = 0.000000001 * CLOCK_TO_MS(CLOCK() - start).count();
	t = time;
	//cout << "Consistent gaussian blur: " << time << endl;
	//lkk[]core;
	return imageFromBytesRGB(p, width, height);
}

#pragma optimize("", off)
Image negative(Image image, double& t)
{
	unsigned char* p;
	int size;
	unsigned int width, height;
	bytesFromImageRGB(image, p, size, width, height);

	auto start = CLOCK();

	#pragma omp parallel for
	for (int i = 0; i < size; i++)
	{
		p[i] = 255 - p[i];
	}

	double time = 0.000000001 * CLOCK_TO_MS(CLOCK() - start).count();
	t = time;
	//cout << "Consistent negative: " << time << endl;
	return imageFromBytesRGB(p, width, height);
}

int main()
{
	Image* images = new Image[6];
	images[0].loadFromFile("300x300.png");
	images[1].loadFromFile("400x400.png");
	images[2].loadFromFile("500x500.png");
	images[3].loadFromFile("600x600.png");
	images[4].loadFromFile("950x950.png");
	images[5].loadFromFile("2400x2400.png");
	double t;
	gaussianBlurWithIntrinsics(images[5], 7, 5, t);
	cout << t;

	//for (int i = 0; i < 6; i++)
	//{
	//	double fullTime = 0;
	//	for (int j = 0; j < 50; j++)
	//	{
	//		double t = 0;
	//		gaussianBlurWithIntrinsics(images[0], 7, 5, t);
	//		fullTime += t;
	//	}
	//	cout << fullTime / 50<< endl;
	//	//cout << i << ": " << fullTime / 50<< endl;
	//}

	/*RenderWindow window(VideoMode(1920, 1000), "Paradigms");
	while (window.isOpen())
	{
		Event event;
		while (window.pollEvent(event))
		{
			if (event.type == Event::Closed) window.close();
		}

		window.clear(Color::Black);
		drawImage(image1, window, Vector2f(0, 0));
		drawImage(image2, window, Vector2f(950, 0));
		window.display();
	}*/
}