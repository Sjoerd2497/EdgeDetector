//
// Created by Sjoerd de Jonge on 06/10/2020.
//

#ifndef EDGEDETECTOR_IMAGE_H
#define EDGEDETECTOR_IMAGE_H


#include "Matrix.h"

class Image : public Matrix{
private:
    bool grayscale;

public:
    // Class functions:
    Image(int width, int height, int layers, bool grayscale);  // Constructs an image
    void convolve(const Matrix& kernel);
    void blurGaussian(double sigma, int kernel_type);
    void rgbToGrayscale(); // Converts RGB/BGR image to grayscale
    void reformatOrigin(bool isTopDown); // Reformats the origin of the image to be top-down or bottom-up
    Image gradientMagnitude(const Image& im_derivX, const Image& im_derivY);
    double getImageAverage(); // Returns the average value of an image pixel channel

    // Getter for class variable grayscale:
    inline bool isGrayScale() const { return grayscale; } // Returns boolean whether grayscale or not
};


#endif //EDGEDETECTOR_IMAGE_H
