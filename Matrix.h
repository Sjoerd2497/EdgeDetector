//
// Created by Sjoerd de Jonge on 21/07/2020.
//

#ifndef EDGEDETECTOR_MATRIX_H
#define EDGEDETECTOR_MATRIX_H

#include <vector>

class Matrix {
private:
    int width;          // Width of the matrix
    int height;         // Height of the matrix
    int layers;         // In case of a 3D matrix
    std::vector<double> data;

public:
    // Class functions:
    Matrix(int width, int height, int layers);  // Constructs a matrix of width * height * layers populated with 0s
    Matrix constructGaussianKernel(int w, int h, double sigma); // Returns a Gaussian matrix with given sigma
    Matrix constructGaussianKernelDerivative(int w, int h, double sigma, bool respecToX = true);// Returns the derivative
                                                                                                // of a Gaussian matrix
    void matrixToImage();   // Transform the values of the matrix within the 0 - 255 range of an image, while keeping
                            // the relative distances.

    // Getters, these are inline and constant:
    inline int getWidth() const { return width; }
    inline int getHeight() const { return height; }
    inline int getLayers() const { return layers; }
    inline const std::vector<double>& getData() const { return data; } // Returns vector which contains the matrix values

    // Setters:
    void setWidth(int new_width);
    void setHeight(int new_height);
    void setData(int i, double new_data);           // Set a single point on the data vector using index value
    void setData(int x, int y, double new_data);    // Set a single point on the data vector using matrix' x,y values
    void setData(std::vector<double> new_data);     // Set a full vector as the data
};


#endif //EDGEDETECTOR_MATRIX_H
