# Применение фильтров негатива и размытия по гауссу на изображениях

## Обзор

Проект на языке программирования C++ предназначен для применения фильтров негатива и размытия по гауссу к изображениям с использованием различной реализации алгоритмов.

## Способы обработки изображений

1. **Последовательный алгоритм:**
   - Применение фильтров негатива и размытия по гауссу последовательно.

2. **OpenMP:**
   - Использование библиотеки OpenMP для параллелизации вычислений.

3. **OpenCL:**
   - Вычисление фильтров на GPU с использованием библиотеки OpenCL.

4. **Векторизация (AVX и intrin.h):**
   - Применение векторизации с использованием AVX инструкций и библиотеки intrin.h.

## Эффективность

Были проведены тесты эффективности каждого способа обработки изображений на различных размерах картинок. Результаты показали, что для малых изображений наилучшей эффективностью обладает последовательный алгоритм, в то время как на больших изображениях вычисления на GPU с использованием OpenCL показывают лучшие результаты.

---

# Image Filters: Negative and Gaussian Blur Application

## Overview

This C++ project is designed to apply negative and Gaussian blur filters to images using various algorithm implementations.

## Image Processing Methods

1. **Sequential Algorithm:**
   - Applying negative and Gaussian blur filters sequentially.

2. **OpenMP:**
   - Utilizing the OpenMP library for parallelizing computations.

3. **OpenCL:**
   - Computing filters on the GPU using the OpenCL library.

4. **Vectorization (AVX and intrin.h):**
   - Applying vectorization using AVX instructions and the intrin.h library.

## Efficiency

Efficiency tests were conducted for each image processing method on various image sizes. The results showed that for small images, the sequential algorithm performs the best, while for larger images, GPU computations using OpenCL exhibit superior performance.
