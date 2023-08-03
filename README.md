# Lane Detection and Distance Estimation with a Monocular Camera

[![Python](https://img.shields.io/badge/Python-3.7%20or%20later-blue.svg)](https://www.python.org/downloads/)

This repository houses all the scripts for the implementation of the computer vision project titiled, "Lane Detection and Distance Estimation with a Monocular Camera." The primary goal is to detect lanes on the road and estimate the distance of the camera from a traffic sign using computer vision techniques. The project uses OpenCV, NumPy, and Matplotlib libraries in Python to process the images or video frames captured by the camera.

## Members of the group:

This project has been carried out by:

Moses Chuka Ebere

Biram Bawo

## Installation

To run this project on your local machine, follow these steps:

1. Clone the repository to your local machine.

2. Make sure you have Python installed on your system. This project is compatible with Python 3.x.

3. Install the required dependencies using the following command:

```
pip install opencv-python numpy matplotlib pandas scipy
```


## Usage

To use this project, follow these steps:

1. Navigate to the project directory.

2. Run the main script by executing the following command:

```
python Main.py
```

3. The script will open the camera feed and start detecting lanes and traffic signs. The output will display the detected lanes and the estimated distance of the camera from the traffic sign.

4. To stop the video feed, press the 'q' key.

## Methodology

The project follows the following steps to perform lane detection and distance estimation:

1. **Object Detection:** The script uses Haar Cascades to detect traffic signs in the camera feed. Detected traffic signs are marked with rectangles and the distance from the camera to the sign is displayed.

2. **Image Preprocessing:** The script applies Gaussian blur and converts the frames to grayscale to reduce noise and improve edge detection.

3. **Edge Detection:** The Canny edge detection algorithm is applied to find edges in the image.

4. **Region of Interest (ROI) Extraction:** The script defines a trapezoidal region of interest, which corresponds to the two lanes of interest on the road. The algorithm masks the image to only consider the region of interest for lane detection.

5. **Lane Detection:** Hough Line Transform is used to extract lines from the ROI. The lines are separated into left and right lanes based on their slopes.

6. **Data Analysis:** Statistical analysis is performed on the slopes and intercepts of the left and right lanes to identify and eliminate outliers. The Interquartile Range (IQR) and skewness are used for this purpose.

7. **Lane Construction:** The final mean slopes and intercepts of the left and right lanes are used to construct the lane lines from the bottom to three-fifths of the image height.

8. **Visualization:** The detected lanes and traffic signs are visualized on the camera feed.

## Video Output

The project generates a video file named "Result.mp4" in the project directory. The video shows the camera feed with detected lanes and traffic signs.

## Report
For further details and visualizations, please refer to the [report](Report/Moses%20Chuka%20Ebere%20-%20EE%20417%20-%20Term%20Project%20Report.pdf).

## Conclusion

In conclusion, this project successfully implements lane detection and distance estimation using a monocular camera. By applying computer vision techniques and statistical analysis, we achieved accurate lane detection and reliable distance estimation from traffic signs. The project's potential applications in autonomous driving and driver-assistance systems make it a valuable contribution to the field of computer vision. Contributions and feedback from the community are welcome to further enhance and refine the implementation. Thank you for using our lane detection and distance estimation project. Happy coding!
