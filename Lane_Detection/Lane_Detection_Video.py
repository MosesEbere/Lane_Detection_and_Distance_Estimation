import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
import pandas as pd

def smooth(img):
    gaussian_filtering = cv2.GaussianBlur(img, (3,3), 0)
    return gaussian_filtering

def convert_to_grayscale(img):
    BW = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return BW

def detect_edges(img):
    # Apply Hysteresis Thresholding: H = 150; L = 50.
    e_image = cv2.Canny(img, 50, 150)
    return e_image

def ROI(img):
    # Find the total number of rows and columns of the image.
    row = img.shape[0]
    column = img.shape[1]
    # Create a trapezoidal region of interest that encompases the two lanes of interest.
    r_o_i = np.array([[(300, row), (1000, row), (600, 300), (550, 300)]])
    # Create a mask (fully black) with the same dimension as the original image.
    mask = np.zeros_like(img)
    # Create a white region - the same shape as our region of interest - out of the mask.
    cv2.fillConvexPoly(mask, r_o_i, 255)
    # Implement a bitwise AND operation on the mask and the image to carve out the region of interest
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def apply_threshold(img):
    # Apply a threshold on the region of interest to eliminate pixels that don't represent the lane.
    # Since the lane is almost white, we can set a lower threshold not too far from 255.
    _, result = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY)
    return result

def find_lines(img):
    # Use Hough Transform to extract lines in the image.
    # Carefully discretize the Hough Space to optimze accuracy.
    # If the precision is too low, there'll be lots of false positives.
    # If the precision is too high, there'll be lots of false negatives.
    # Set a threshold to determine which lines should be drawn.
    # Reject any lines below 40, and join lines with a gap of 5 between them.
    #lines = cv2.HoughLinesP(img, 1, np.pi/180, 30, maxLineGap=200)
    lines = cv2.HoughLinesP(img, 1, np.pi/180, 70, np.array([]), minLineLength=20, maxLineGap=50)
    return lines

def construct_the_lanes(img, slope_and_intercept):
    final_slope, final_intercept = slope_and_intercept
    # Apply the equation of a line, y = mx+c, to construct the left and right lane using the mean slope and intercept.
    y1 = img.shape[0] # Since the first component contains the image rows (max = bottom)
    y2 = int(y1*(3/5)) # The lanes would extend from the bottom to 3/5 of the image height.
    x1 = int((y1 - final_intercept)/final_slope)
    x2 = int((y2 - final_intercept)/final_slope)
    # Return the coordinates as a 1D array
    return np.array([x1, y1, x2, y2])

def data_analysis_of_lines(img, lines):
    left_lanes_array = []
    right_lanes_array = []
    left_slope_array = []
    right_slope_array = []
    left_intercept_array = []
    right_intercept_array = []
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        # Obtain the slope and intercept of each line using linear regression
        x = [x1, x2]
        y = [y1, y2]
        slope, intercept, r_value, p_value, std_err = linregress(x, y)
        # Recall that the x-y plane of an image is different. The origin is at the top-left corner.
        # Therefore, lines on the left would have a negative slope (+ve for lines on the right).
        # Separate the left and right lines
        if slope < 0:
            left_lanes_array.append((slope, intercept))
            left_slope_array.append(slope)
            left_intercept_array.append(intercept)
        else:
            right_lanes_array.append((slope, intercept))
            right_slope_array.append(slope)
            right_intercept_array.append(intercept)

    # Create Pandas Dataframes for the arrays obtained above for statistical calculations.
    df_left = pd.DataFrame({'slope': left_slope_array,
                   'intercept': left_intercept_array})

    df_right = pd.DataFrame({'slope': right_slope_array,
               'intercept': right_intercept_array})

    ####### DATA ANALYSIS ON THE HOUGHLINES FOUND TO CHECK AND ELIMINATE OUTLIERS. #######
    ###### USING MEASURES OF STATISTICAL DISPERSION
    # This is used to find the interquartile range.
    # Q1 = df_left.quantile(0.25)
    # Q3 = df_left.quantile(0.75)
    # IQR = Q3 - Q1
    # print(IQR)

    # Use the interquartile range to check which array components are OUTLIERS.
    # True denotes the presence of an OUTLIER.
    # print((df_left < (Q1 - 1.5 * IQR)) | (df_left > (Q3 + 1.5 * IQR)))

    # Use the skew of the array to check which array components are OUTLIERS.
    # print(df_left['slope'].skew())
    # print(df_left['slope'].describe())
    # print(df_left['intercept'].describe())
    #
    # print(df_right['slope'].describe())
    # print(df_right['intercept'].describe())

    ###### THE FOLLOWING COULD BE USED TO VISUALLY IDENTIFY OUTLIERS.
    ### Box Plot
    # plt.boxplot(df_left["slope"])
    # plt.show()
    # plt.boxplot(df_left["intercept"])
    # plt.show()
    # plt.boxplot(df_right["slope"])
    # plt.show()
    # plt.boxplot(df_right["intercept"])
    # plt.show()

    ### Scatter Plot
    # Left
    # fig, ax = plt.subplots(figsize=(12,6))
    # ax.scatter(df_left['slope'], df_left['intercept'])
    # ax.set_xlabel('slope')
    # ax.set_ylabel('intercept')
    #plt.show()

    # Right
    # figg, aax = plt.subplots(figsize=(12,6))
    # aax.scatter(df_right['slope'], df_right['intercept'])
    # aax.set_xlabel('slope')
    # aax.set_ylabel('intercept')
    #plt.show()

    ### USE THE INTERQUARTILE RANGE TO ELIMINATE OUTLIERS
    df_left["slope"] = np.where(df_left["slope"] < df_left['slope'].quantile(0.25), df_left['slope'].quantile(0.25),df_left['slope'])
    df_left["slope"] = np.where(df_left["slope"] > df_left['slope'].quantile(0.75), df_left['slope'].quantile(0.75),df_left['slope'])

    df_left["intercept"] = np.where(df_left["intercept"] < df_left['intercept'].quantile(0.25), df_left['intercept'].quantile(0.25),df_left['intercept'])
    df_left["intercept"] = np.where(df_left["intercept"] > df_left['intercept'].quantile(0.75), df_left['intercept'].quantile(0.75),df_left['intercept'])

    df_right["slope"] = np.where(df_right["slope"] < df_right['slope'].quantile(0.25), df_right['slope'].quantile(0.25),df_right['slope'])
    df_right["slope"] = np.where(df_right["slope"] > df_right['slope'].quantile(0.75), df_right['slope'].quantile(0.75),df_right['slope'])

    df_right["intercept"] = np.where(df_right["intercept"] < df_right['intercept'].quantile(0.25), df_right['intercept'].quantile(0.25),df_right['intercept'])
    df_right["intercept"] = np.where(df_right["intercept"] > df_right['intercept'].quantile(0.75), df_right['intercept'].quantile(0.75),df_right['intercept'])

    # print(df_left['slope'].describe())
    # print(df_left['intercept'].describe())
    #
    # print(df_right['slope'].describe())
    # print(df_right['intercept'].describe())

    # Ensure that lists are not empty
    if len(right_lanes_array) == len(left_lanes_array) == 0:
        return np.array([])
    if len(left_lanes_array) == 0:
        ### Using mean
        right_slope = df_right["slope"].mean()
        right_intercept = df_right["intercept"].mean()

        ### Using median # right_slope = df_right['slope'].median() # right_intercept = df_right['intercept'].median()
        right = np.array([right_slope, right_intercept])
        right_line = construct_the_lanes(img, right)
        return np.array([right_line])
    elif len(right_lanes_array) == 0:
        ### Using mean
        left_slope = df_left['slope'].mean()
        left_intercept = df_left['intercept'].mean()

        ### Using median # left_slope = df_left['slope'].median() # left_intercept = df_left['intercept'].median()
        left = np.array([left_slope, left_intercept])
        left_line = construct_the_lanes(img, left)
        return np.array([left_line])

    left_slope = df_left['slope'].mean()
    left_intercept = df_left['intercept'].mean()
    left = np.array([left_slope, left_intercept])

    right_slope = df_right["slope"].mean()
    right_intercept = df_right["intercept"].mean()
    right = np.array([right_slope, right_intercept])

    left_line = construct_the_lanes(img, left)
    right_line = construct_the_lanes(img, right)
    return np.array([left_line, right_line])

def display_lines(img, lines):
    #line_image = np.zeros_like(img)
    if lines is not None:
        for line  in lines:
            x1, y1, x2, y2 = line.reshape(4)
            # Draw a line with thickness = 10
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 10)
    return img

# img = cv2.cvtColor(cv2.imread('test_image.jpg'), cv2.COLOR_BGR2RGB)
# #img = cv2.imread('12.png')
# image_frame = np.copy(img)
# smoothed_image = smooth(image_frame)
# gray_image = convert_to_grayscale(smoothed_image)
# edge_image = detect_edges(gray_image)
# ROI_image = ROI(gray_image)
# threshold_result = apply_threshold(ROI_image)
# detected_lines = find_lines(threshold_result)
#
# final_lines = data_analysis_of_lines(image_frame, detected_lines)
# #     combo_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1)
# #line_image = display_lines(image_frame, detected_lines)
# line_image = display_lines(image_frame, final_lines)
# # cv2.imshow("result", line_image)
# # cv2.waitKey(0)
# #plt.imshow(gray_image, cmap = "gray")
# plt.imshow(line_image)
# plt.show()

# Use ctrl + / to comment out blocks of code

# Create a capture variable for the video
video = cv2.VideoCapture("test2.mp4")
# Read each frame once the video is opened
while(video.isOpened()):
    # Create an empty variable for the boolean value and frame for each frame
    not_used, image_frame = video.read()
    if image_frame is None:
            break
    image_frame = image_frame.astype('uint8')

    smoothed_image = smooth(image_frame)
    gray_image = convert_to_grayscale(smoothed_image)
    edge_image = detect_edges(gray_image)
    ROI_image = ROI(gray_image)
    threshold_result = apply_threshold(ROI_image)
    # plt.imshow(threshold_result)
    # plt.show()
    detected_lines = find_lines(threshold_result)

    final_lines = data_analysis_of_lines(image_frame, detected_lines)
    #     combo_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1)
    #line_image = display_lines(image_frame, detected_lines)
    line_image = display_lines(image_frame, final_lines)
# cv2.imshow("result", line_image)
# cv2.waitKey(0)
#plt.imshow(gray_image, cmap = "gray")
# plt.imshow(line_image)
# plt.show()
#     smoothed_image = smooth(image_frame)
#     gray_image = convert_to_grayscale(smoothed_image)
#     ROI_image = ROI(gray_image)
#     edge_image = detect_edges(ROI_image)
#     detected_lines = find_lines(edge_image)
#
#     mean_lines = mean_slope_and_intercept(frame, lines)
#     line_image = display_lines(frame, mean_lines)
#     combo_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1)
    cv2.imshow("result", line_image)
    # Wait 1ms between each frame
    # Use the q button to stop the video
    if cv2.waitKey(1) == ord('q'):
        break
video.release()
cv2.destroyAllWindows()
