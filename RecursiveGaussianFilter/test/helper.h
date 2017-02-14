#ifndef HELPER_H
#define HELPER_H

#include <iostream>
#include <string>
#include <cstdlib>

#include <opencv2/core/core.hpp>  
#include <opencv2/highgui/highgui.hpp> 

using namespace std;

template<typename T>
bool load_img(T **data, size_t &rows, size_t &cols)
{
    string filename = "../data/cameraman.tif";
    cv::Mat img = cv::imread(filename, CV_LOAD_IMAGE_GRAYSCALE);
    rows = img.rows;
    cols = img.cols;

    *data = new float[rows * cols];

    for(int i = 0; i < rows * cols; ++i)
    {
        (*data)[i] = (T)img.data[i];
    }
    return rows > 0 && cols > 0;
}

template<typename T>
bool random_img(T **data, size_t rows, size_t cols)
{
    *data = new float[rows * cols];

    for(int i = 0; i < rows * cols; ++i)
    {
        (*data)[i] = (T)(rand() % 256);
    }
    return true;
}

template<typename T>
void show_image(const T *data, size_t rows, size_t cols, string title = "image0")
{
    cv::Mat img(rows, cols, CV_8U);

    for(int i = 0; i < rows * cols; ++i)
    {
        img.data[i] = (unsigned int)data[i];
    }

    cv::namedWindow( title, CV_WINDOW_AUTOSIZE );
    cv::imshow( title, img );
}

void show_all();

#endif