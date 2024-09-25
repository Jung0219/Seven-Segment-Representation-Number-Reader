# Digit Detection from Seven-Segment Displays

## Overview

This project involves the detection and recognition of digits from images of seven-segment displays using OpenCV and Python. The implementation is part of my personal study of OpenCV, focusing on image processing techniques.

## Features
# Digit Detection Process

## Image Loading
The script loads an image containing a seven-segment display.

## Image Preprocessing
- The image is converted to RGB format and then to grayscale.
- A binary threshold is applied to isolate the segments.

## Contour Detection
- Contours are detected in the thresholded image.
- The largest contour (the display area) is identified.

## Perspective Transformation
- A perspective transform is applied to obtain a top-down view of the digit display.

## Digit Detection
- Morphological operations are performed to fill gaps in the detected digits.
- Contours of individual digits are extracted and filtered based on size.

## Digit Recognition
- Each detected digit is analyzed by checking the segments of the seven-segment display to determine its value.
- The recognized digits are drawn on the output image.

## Output
The result is saved as `result.jpg`, showing the detected digits on the original image.


## Technologies Used

- Python 3.10
- OpenCV
- NumPy
- matplotlib
