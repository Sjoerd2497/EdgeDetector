//
// Created by Sjoerd de Jonge on 21/07/2020.
//

#include "Matrix.h"
#include <cmath>
#include <utility>
#include <vector>
#include <iostream>
#include <algorithm>

const double pi = 3.1415926535897;

/// Construct a matrix with int width, int height and int layers and default values 0.
Matrix::Matrix(int width, int height, int layers) {
    // Evaluate input; make sure size > 0:
    if(width <= 0 || height <=0 || layers <=0){
        std::cout << "Error: constructor. Width, height and/or layers of the matrix must be larger than 0. Returning 1x1x1 matrix."
        << std::endl;
        width = 1;
        height = 1;
        layers = 1;
    }
    this->width = width;
    this->height = height;
    this->layers = layers;
    data.resize( width * height * layers, 0);
}

/// Returns a Gaussian kernel (=a matrix) of w x h with sigma. Width and height must always be odd.
/// Matrix can be 1D (1xN or Nx1) or 2D (MxN or MxM).
Matrix Matrix::constructGaussianKernel(int w, int h, double sigma) {
    // Evaluate input; make sure kernel has odd size and size > 0:
    if(w % 2 == 0){
        ++w;
        std::cout << "Warning: Width of the Gaussian matrix must be odd. Using width = " << w << " instead."
                  << std::endl;
    }
    if(h % 2 == 0){
        ++h;
        std::cout << "Warning: Height of the Gaussian matrix must be odd. Using height = " << h << " instead."
                  << std::endl;
    }
    if(w <= 0 || h <= 0){
        std::cout << "Error: Matrix::constructGaussianKernel. Width and/or height of the Gaussian matrix must be larger "
                     "than 0. Returning empty 1x1 matrix." << std::endl;
        return Matrix(1, 1, 1);
    }
    Matrix g(w, h, 1);         // The matrix that will be returned
    int x_min = (int) -floor(w/2);     // The minimum x value, for a kernel width of 5 we would get -2
    int y_min = (int) -floor(h/2);     // The minimum y value
    double sum = 0;         // The sum of all elements in the matrix
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            // For every element of the matrix:
            int xval = x_min + x;
            int yval = y_min + y;

            // Determine whether matrix is 1D-horizontal, 1D-vertical or 2D and construct kernel:
            if (w == 1) {
                // 1D-vertical Gaussian:
                g.data[y * w + x] =
                        (1 / sqrt(2 * pi * pow(sigma, 2))) * exp(-(pow(yval, 2)) / (2 * pow(sigma, 2)));
                sum += g.data[y * w + x];
            } else if (h == 1) {
                // 1D-horizontal Gaussian:
                g.data[y * w + x] =
                        (1 / sqrt(2 * pi * pow(sigma, 2))) * exp(-(pow(xval, 2)) / (2 * pow(sigma, 2)));
                sum += g.data[y * w + x];
            } else {
                // 2D Gaussian:
                // Calculating the value for the Gaussian, using 1/(2*pi*sigma^2) * e^((x^2 + y^2)/2*sigma^2):
                g.data[y * w + x] = (1 / (2 * pi * pow(sigma, 2.0))) *
                                      exp(-(pow(xval, 2.0) + pow(yval, 2.0)) / (2 * pow(sigma, 2.0)));
                sum += g.data[y * width + x];
            }
        }
    }
    // Normalizing the created matrix so the sum of all elements is 1. This is to ensure the image that will be convolved
    // by this kernel will retain its average brightness.
    // See also: https://en.wikipedia.org/wiki/Kernel_(image_processing)#Normalization
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            g.data[y*w+x]/=sum;   // Dividing each element of the matrix by the sum of all elements
        }
    }
    return g;
}


/// Returns the derivative of a Gaussian kernel (matrix) of w x h with sigma. Width and height must always be odd.
/// Matrix can be 1D (1xN or Nx1) or 2D (MxN or MxM).
/// In case of a 2D matrix, the bool respectToX determines whether the derivative with respect to X is or with respect
/// to Y is taken.
/// Bool respectToX is true by default!
Matrix Matrix::constructGaussianKernelDerivative(int w, int h, double sigma, bool respectToX) {
    // Evaluate input; make sure kernel has odd size and size > 0:
    if(w % 2 == 0){
        ++w;
        std::cout << "Warning: Matrix::constructGaussianKernelDerivative. Width of the Gaussian matrix must be odd. "
                     "Using width = " << w << " instead."
                  << std::endl;
    }
    if(h % 2 == 0){
        ++h;
        std::cout << "Warning: Matrix::constructGaussianKernelDerivative. Height of the Gaussian matrix must be odd. "
                     "Using height = " << h << " instead."
                  << std::endl;
    }
    if(w <= 0 || h <= 0){
        std::cout << "Error: Matrix::constructGaussianKernelDerivative. Width and/or height of the Gaussian matrix must "
                     "be larger than 0. Returning empty 1x1 matrix." << std::endl;
        return Matrix(1, 1, 1);
    }
    Matrix g(w, h, 1);         // The matrix that will be returned
    int x_min = (int) -floor(w/2);     // The minimum x value, a kernel width of 5 gives -2 as minimum
    int y_min = (int) -floor(h/2);     // The minimum y value
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            // For every element of the matrix:
            int xval = x_min + x;
            int yval = y_min + y;

            // Determine whether matrix is 1D-horizontal, 1D-vertical or 2D and construct kernel:
            if (w == 1) {
                // 1D-vertical Gaussian:
                // Formula used is: (y / (sigma^3*sqrt(2*pi)) * exp((-y^2)/(2*sigma^2))
                // This formula, and the other derivate formulas below, should start with a negative sign, but due to the
                // different way coordinates are handled in images vs graphs, this is ommited to compensate for the
                // difference.
                g.data[y*width+x]= (yval)/(pow(sigma, 3)*sqrt(2*pi))*exp(-(pow(yval, 2))/(2*pow(sigma, 2)));
            } else if (h == 1) {
                // 1D-horizontal Gaussian:
                // Formula used is: (x / (sigma^3*sqrt(2*pi)) * exp((-x^2)/(2*sigma^2))
                g.data[y*width+x]= (xval)/(pow(sigma, 3)*sqrt(2*pi))*exp(-(pow(xval, 2))/(2*pow(sigma, 2)));
            } else {
                // 2D Gaussian:
                // Calculating the value for the derivative of a Gaussian
                if (respectToX){
                    // Formula used is: (x)/(2*pi*sigma^4) * exp(-(x^2+y^2)/(2*sigma^2))
                    g.data[y*width+x]=(xval)/(2*pi*pow(sigma,4))*exp(-(pow(xval,2)+pow(yval,2))/(2*pow(sigma,2)));
                } else {
                    // Formula used is: -(y)/(2*pi*sigma^4) * exp(-(x^2+y^2)/(2*sigma^2))
                    g.data[y*width+x]=(yval)/(2*pi*pow(sigma,4))*exp(-(pow(xval,2)+pow(yval,2))/(2*pow(sigma,2)));
                }
            }
        }
    }

    // Normalizing is not needed for the derivative of a Gaussian, since Matrix::matrixToImage() will be used for this
    // purpose.
    return g;
}

/// Transform any matrix into the byte image range (0 to 255). Similar to MATLAB's mat2gray, which source code was
/// studied to make this function in C++.
void Matrix::matrixToImage() {
    // Get the max and min value of the matrix:
    double max = *std::max_element(std::begin(data), std::end(data));
    // (std::max_element() returns an iterator so the dereference operator is used)
    double min = *std::min_element(std::begin(data), std::end(data));

    // Delta is the weight factor for each pixel:
    double delta = 255/(max-min);

    // Range-based for loop through the data of the matrix:
    for (double & i : data){
        i *= delta;
        i += -min*delta;

        // Make sure all values are between 0-255:
        if (i < 0){
            i = 0;
        } else if (i > 255){
            i = 255;
        }
    }
}

/// Change the private integer 'width' of the matrix and resize matrix accordingly.
void Matrix::setWidth(int new_width) {
    if (0 < new_width){
        width = new_width;
        data.resize(width * height);
    } else{
        std::cout << "Error: Matrix::setWidth. Values equal to or smaller than 0 are not allowed for setting matrix width."
        << std::endl;
    }
}

/// Change the private integer 'height' of the matrix and resize matrix accordingly.
void Matrix::setHeight(int new_height) {
    if (0 < new_height){
        height = new_height;
        data.resize(width * height);
    } else{
        std::cout << "Error: Matrix::setHeight. Values equal to or smaller than 0 are not allowed for setting matrix height."
        << std::endl;
    }
}

/// Change the private vector<double> 'data[i]' of the matrix.
void Matrix::setData(int i, double new_data) {
    if( (0 <= i) && (i <= (data.size()-1)) ){
        data[i] = new_data;
    }
    else{
        std::cout << "Error: Matrix::setData. Cannot set matrix data. Index out of range. Max index size is: " << data.size()-1
        << std::endl;
    }
}

/// Change the private vector<double> 'data[y * getWidth() + x]' of the matrix.
void Matrix::setData(int x, int y, double new_data) {
    --x; // Decrease x by 1, because vector index starts at 0
    --y; // Decrease y by 1, because vector index starts at 0
    if( 0 <= (y * getWidth() + x) && (y * getWidth() + x) <= (data.size()-1) ){
        data[y * getWidth() + x] = new_data;
    }
    else{
        std::cout << "Error: Matrix::setData. Cannot set matrix data. Index out of range. Max index size is: " << data.size()-1
        << std::endl;
    }
}

/// Set a full vector as the data
void Matrix::setData(std::vector<double> new_data){
    if (new_data.size() != width*height*layers){
        std::cout << "Error: Matrix::setData. Cannot set matrix data, vector does not match matrix size." << std::endl;
    }
    else{
        data = new_data;
    }
}





