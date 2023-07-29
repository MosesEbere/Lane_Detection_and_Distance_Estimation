import cv2
import numpy as np
import matplotlib.pyplot as plt
import statistics
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
    r_o_i = np.array([[(70, row), (column, row), (column, 1020), (970, 635), (690, 638)]])
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
    _, result = cv2.threshold(img, 170, 255, cv2.THRESH_BINARY)
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

def display_lines(img, lines):
    #line_image = np.zeros_like(img)
    if lines is not None:
        for line  in lines:
            #x1, y1, x2, y2 = line[0]
            #print(line)
            x1, y1, x2, y2 = line.reshape(4)
            # Draw a blue line (BGR) with thickness = 10
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 10)
    return img

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
    # # This is used to find the interquartile range.
    # Q1 = df_left.quantile(0.25)
    # Q3 = df_left.quantile(0.75)
    # IQR = Q3 - Q1
    # print("Interquartile Range for the left lines \n", IQR)
    #
    # Q1_ = df_right.quantile(0.25)
    # Q3_ = df_right.quantile(0.75)
    # IQR_ = Q3_ - Q1_
    # print("Interquartile Range for the right lines \n", IQR_)
    #
    # # Use the interquartile range to check which array components are OUTLIERS.
    # # True denotes the presence of an OUTLIER.
    # print((df_left < (Q1 - 1.5 * IQR)) | (df_left > (Q3 + 1.5 * IQR)))
    # print((df_right < (Q1_ - 1.5 * IQR_)) | (df_right > (Q3_ + 1.5 * IQR_)))
    #
    # # Use the skew of the array to check which array components are OUTLIERS.
    # print("Skew for the left lines", df_left['slope'].skew())
    # print(df_left['slope'].describe())
    # print(df_left['intercept'].describe())
    #
    # print("Skew for the right lines", df_right['slope'].skew())
    # print(df_right['slope'].describe())
    # print(df_right['intercept'].describe())
    #
    # ###### THE FOLLOWING COULD BE USED TO VISUALLY IDENTIFY OUTLIERS.
    # ## Box Plot
    # plt.boxplot(df_left["slope"])
    # plt.title('Box Plot for the Slopes of the Left Lines')
    # plt.show()
    # plt.boxplot(df_left["intercept"])
    # plt.title('Box Plot for the Intercepts of the Left Lines')
    # plt.show()
    # plt.boxplot(df_right["slope"])
    # plt.title('Box Plot for the Slopes of the Right Lines')
    # plt.show()
    # plt.boxplot(df_right["intercept"])
    # plt.title('Box Plot for the Intercepts of the Right Lines')
    # plt.show()
    #
    # ### Scatter Plot
    # # Left
    # fig, ax = plt.subplots(figsize=(12,6))
    # ax.scatter(df_left['slope'], df_left['intercept'])
    # ax.set_xlabel('slope')
    # ax.set_ylabel('intercept')
    # ax.set_title('Scatter Plot for the Left Lines')
    # plt.show()
    #
    # # Right
    # figg, aax = plt.subplots(figsize=(12,6))
    # aax.scatter(df_right['slope'], df_right['intercept'])
    # aax.set_xlabel('slope')
    # aax.set_ylabel('intercept')
    # aax.set_title('Scatter Plot for the Right Lines')
    # plt.show()

    ### USE THE INTERQUARTILE RANGE TO ELIMINATE OUTLIERS
    df_left["slope"] = np.where(df_left["slope"] < df_left['slope'].quantile(0.25), df_left['slope'].quantile(0.25),df_left['slope'])
    df_left["slope"] = np.where(df_left["slope"] > df_left['slope'].quantile(0.75), df_left['slope'].quantile(0.75),df_left['slope'])

    df_left["intercept"] = np.where(df_left["intercept"] < df_left['intercept'].quantile(0.25), df_left['intercept'].quantile(0.25),df_left['intercept'])
    df_left["intercept"] = np.where(df_left["intercept"] > df_left['intercept'].quantile(0.75), df_left['intercept'].quantile(0.75),df_left['intercept'])

    df_right["slope"] = np.where(df_right["slope"] < df_right['slope'].quantile(0.25), df_right['slope'].quantile(0.25),df_right['slope'])
    df_right["slope"] = np.where(df_right["slope"] > df_right['slope'].quantile(0.75), df_right['slope'].quantile(0.75),df_right['slope'])

    df_right["intercept"] = np.where(df_right["intercept"] < df_right['intercept'].quantile(0.25), df_right['intercept'].quantile(0.25),df_right['intercept'])
    df_right["intercept"] = np.where(df_right["intercept"] > df_right['intercept'].quantile(0.75), df_right['intercept'].quantile(0.75),df_right['intercept'])

    # Show the new dataframes
    # print(df_left['slope'].describe())
    # print(df_left['intercept'].describe())
    #
    # print(df_right['slope'].describe())
    # print(df_right['intercept'].describe())

    ### These conditional statements are used to check for frames without detections and skip them to avoid errors.
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

###########################
# Used to test an image or a single video frame.
# Comment out from here till line 225 and "uncomment" lines 229 to 252
img = cv2.cvtColor(cv2.imread('1.jpg'), cv2.COLOR_BGR2RGB)
image_frame = np.copy(img)
smoothed_image = smooth(image_frame)
gray_image = convert_to_grayscale(smoothed_image)
edge_image = detect_edges(gray_image)
ROI_image = ROI(edge_image)
threshold_result = apply_threshold(ROI_image)
detected_lines = find_lines(threshold_result)

final_lines = data_analysis_of_lines(image_frame, detected_lines)
line_image = display_lines(image_frame, final_lines)
plt.imshow(line_image)
# plt.title('Hough Lines after Post-processing (using Mean)')
plt.show()

# Use ctrl + / to comment out blocks of code

######################
# Create a capture variable for the video
# def main():
#     windowname = "Detected Lanes"
#     cv2.namedWindow(windowname)
#
#     # Create a capture variable for the video
#     video = cv2.VideoCapture("Road.mp4")
#
#     filename = "Result.mp4v"
#     codec = cv2.VideoWriter_fourcc(*'XVID')
#     framerate = video.get(cv2.CAP_PROP_FPS)
#     # VideoOutPut = cv2.VideoWriter(filename, codec, framerate, resolution)
#
#     if video.isOpened():
#         ret, frame = video.read()
#     else:
#         ret = False
#
#     image_frame_array = []
#     # Read each frame once the video starts
#     while ret:
#         # Create an empty variable for the boolean value and frame for each frame
#         ret, image_frame = video.read()
#         if image_frame is None:
#                 break
#         image_frame = image_frame.astype('uint8')
#
#         smoothed_image = smooth(image_frame)
#         gray_image = convert_to_grayscale(smoothed_image)
#         edge_image = detect_edges(gray_image)
#         ROI_image = ROI(gray_image)
#         threshold_result = apply_threshold(ROI_image)
#         detected_lines = find_lines(threshold_result)
#
#         final_lines = data_analysis_of_lines(image_frame, detected_lines)
#         line_image = display_lines(image_frame, final_lines)
#
#         # Obtain the size of one frame and store all frames in an array
#         height, width = line_image.shape[:2]
#         resolution = (width, height)
#         image_frame_array.append(line_image)
#
#         cv2.imshow(windowname, line_image)
#         # Wait 1ms between each frame
#         # Use the q button to stop the video
#         if cv2.waitKey(1) == ord('q'):
#             break
#
#     # write the video
#     VideoOutPut = cv2.VideoWriter(filename, codec, framerate, resolution)
#
#     for i in range(len(image_frame_array)):
#         # writing to a image array
#         VideoOutPut.write(image_frame_array[i])
#
#     cv2.destroyAllWindows()
#     VideoOutPut.release()
#     video.release()
#
# if __name__ == "__main__":
#     main()
