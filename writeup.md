## Writeup

---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./output_images/find_corner.png "Find Corners"
[image2]: ./output_images/undistort_chessboard.png "Undistort Chessboard Example"
[image3]: ./output_images/undistort_test_images.png "Undistort Test Images"
[image4]: ./output_images/test_image_hls_space.png "Test Images in HLS color space"
[image5]: ./output_images/sobel_abs_test_images.png "Sobel Thresh Grads in X Direction"
[image6]: ./output_images/sobel_mag_test_images.png "Sobel Thresh Grads in Magnititude"
[image7]: ./output_images/sobel_direction_test_images.png "Sobel Thresh Grads in Direction"
[image8]: ./output_images/sobel_combined_test_images.png "Sobel Thresh Grads Combined"
[image9]: ./output_images/Perpective_transform_marked.png "Perspective Transformation With Marked Points"
[image10]: ./output_images/perpective_transformed_test_images.png "Perspective Transformation for All Test Images"
[image11]: ./output_images/perspective_transformed_combind_grad_test_images.png "Perspective Transformation for All Test Images Combined Grads"
[image12]: ./output_images/detect_lane_area_test_images.png "Detected Lane Areas for All Test Images Side by Side"
[image13]: ./output_images/final_output_test_images.png "Final output of Processed Test Images"
[image14]: ./output_images/challenge_image.png "Modified Challenge Image Pipeline"
[video1]: ./project_video_output.mp4 "Video Output"
[video2]: ./challenge_video_output.mp4 "Video Output Based on S Channel"
[video3]: ./challenge_video_output_gray.mp4 "Video Output Based on Gray Channel"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the first code cell of the IPython notebook located in "./project.ipynb"

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][image1]
![alt text][image2]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to all test images:
![alt text][image3]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image. Please refer "./project.ipynb" code block 2~8. 

First I convert the image to HLS color space. I see S Channel always contains lane line information regardless light.
![alt text][image4]

Second, I use sobel threshholded gradients absolute value on X orientation on S Channel: 
![alt text][image5]

Third, I use sobel threshholded gradients magnititude on S Channel: 
![alt text][image6]

Then, I use sobel threshholded gradients direction on S Channel: 
![alt text][image7]

Finally, I combined all above altogether: 
![alt text][image8]


#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `warper()`, which appears in the 9th code cell of the IPython notebook).  The `warper()` function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points.  I chose the hardcode the source and destination points in the following manner:

```python
src = np.float32(
    [[(img_size[0] / 2) - 55, img_size[1] / 2 + 100],
    [((img_size[0] / 6) - 10), img_size[1]],
    [(img_size[0] * 5 / 6) + 60, img_size[1]],
    [(img_size[0] / 2 + 55), img_size[1] / 2 + 100]])
dst = np.float32(
    [[(img_size[0] / 4), 0],
    [(img_size[0] / 4), img_size[1]],
    [(img_size[0] * 3 / 4), img_size[1]],
    [(img_size[0] * 3 / 4), 0]])
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 585, 460      | 320, 0        | 
| 203, 720      | 320, 720      |
| 1127, 720     | 960, 720      |
| 695, 460      | 960, 0        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image9]

Then I applied transform to all test images: 
![alt text][image10]

And also applied transform to all test images combined grads images 
![alt text][image11]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

The line detection code could be found at code block 12 in  "./project.ipynb". The algorithm calculates the histogram on the X axis. The maximum two peak values suggest the left lane and right lane, the algorithms then collect the non-zero points contained on those windows and do a polynomial fit to find the line model.  The following picture shows the procedure:

![alt text][image12]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in same code block 12 "./project.ipynb" of by `draw_curvature()`. The function use 
``` python
((1 + (2*fit[0]*y_eval*ym_per_pix + fit[1])**2)**1.5) / np.absolute(2*fit[0])
```
to caculate the curvature. fit the polynomial parameters. y_eval is the maximum Y value and ym_per_pix is to convert pixel to meter.

To find the position to the center, first caculate the middle point of left lane and right lane at max Y. Then find the difference with center. (Assuming the image is from center camera). Convert pixel to meter as well.


#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I applied the pipeline to test images, and here is the result. 

![alt text][image13]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  

Here's a  [project_video_output.mp4](./project_video_output.mp4)

Here's a  [challenge_video_output.mp4](./challenge_video_output.mp4)


---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

It works ok in project_video.mp4, but perform poorly in the challenge_video.mp4.  I saved a frame under "./challenge_images/" and did more analysis, looks previously using S channel alone is not always right. In this case, L channel or gray can detect image more accurately. Refer 

![alt text][image14]

Rerun for the challenge_video with gray channel.

Here's a  [challenge_video_output_gray.mp4](./challenge_video_output_gray.mp4)

Not perfect but it looks better than S channel.

Another optimization may potentially work is that we know the lane width should within some predefined threshhold, we can use this information to eliminate the wrong one and re-calculate with a different algorithm or different color space. It is sort of adaptive optimization.


