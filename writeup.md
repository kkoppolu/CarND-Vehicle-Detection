**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./examples/car_not_car.png
[image2]: ./examples/HOG_example.PNG
[image3]: ./examples/sliding_window.png
[image4]: ./examples/bboxes_and_heat.png
[image5]: ./examples/car_detections.png


## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the third code cell of the IPython notebook. The method extracting HOG features is `get_hog_features`

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:

![alt text][image2]

#### 2. Explain how you settled on your final choice of HOG parameters.

The color channel chosen was `YCrCb` because it separates out intensity component of the image (Y channel) from its color characteristics (chroma channels).
The pixels per cell and cells per block were chosen based on the size of the features we are looking for in the image (the car).
The orientations were chosen as 9 as a good trade off between model fidelity and number of output HOG parameters feeding into the classification model. 

In addition to using HOG features, the following are also used:
- Spatail binning to capture spatial information in the images (relative position of the car)
- Color histogram to capture color features of the car. (The car color will stand out from its background)
 
#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

The model code is available in the function `train`.  
A Grid Search was employed to to choose the best model among the following parameters:

| Kernel  | C  |
| --------|----:|
| linear  | 1   |
| linear  | 10  |
| linear  | 100 |
| poly    | 1   |
| poly    | 10  |
| poly    | 100 |
| rbf     | 1   |
| rbf     | 10  |
| rbf     | 100 |

Based on the grid search, the best classifier turned out to be  
`rbf kernel` with `C` set to `10`. A test accuracy of `0.9944` was achieved.

The features are scaled to zero mean and unit variance and are fed to the classifier in the following order:
- Spatial
- Color
- HOG

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

The sliding window search is implemented using sub-sampling of HOG features. This can be found in the methods `find_cars` and `find_cars_single_param`. They consists of the following steps:

1. A base window size of `64 X 64` is selected.
2. For each scale, a start y position and end y position are selected. This is so that we do not want unwanted features to come up as false positives (like tree tops). 
Aso, further in the horizon, the cars tend to be smaller than closer to the car where they thend to be larger.  
3. The image is resized instead of rescaling the sliding window such that the net effect remains the same.  
4. HOG features are computed for the rescaled image
5. For each patch of the image, HOG features are sub-sampled from the HOG feature array.
6. Spatial and color features are computed for the image patch (64 X 64) in the rescaled image
7. A prediction is made using the extracted features. If the prediction is positive, corresponding pixels are lit in the heat map.
8. The heat map is aggregated across all scales.
9. The aggregated heatmap is averaged over the last `5` values.
10. The heat map is then thresholded to `3`.

After conducting trials on the test images and the test video, the sliding widow search was done for the following parameters:  

| Scale  | Y Start  | Y End  |
| ------:|---------:| ------:|
| 1      | 400      | 480    |
| 1.5    | 400      | 680    |
| 2.5    | 480      | 680    |

The resulting search grid looks like this:  
![alt text][image3]

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

As mentioned previously, the classifier parameters were tuned using `grid search`. The feature selection described previously also contributed to the improved performance of the classifier.
The sliding window scales and y positions were tuned keeping in mind the layout of the image (described in the sliding window approach).  
Here are some example images:

![alt text][image4]
---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected. False posotive detections are minimized by thresholding the heat map so that spurrious detections are eliminated. Moving average of heat maps across previous heat maps (`10`) is used to smooth out the output and reduce jitter.

Here is an example of the overlapping bounding boxes and how they are processed into the video frame:
![Overlapping boxes][image5]
### Here is a test video demonstrating the approach:  

Here's a [Short video with integrated bounding boxes](./test.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

- The pipeline implements a rudimentary smoothing technique of moving average and thresholding. A smarter smoothing algorithm with better false positive rule out can be thought of.  
- Once a vehicle is detected, the pipeline does not keep track of it. It only keeps track of the heat map from the previous frames. Fine grained memory across frames will lead to a better false positive elimination.
- The pipeline seems to be having difficulty with a white car on a bright day.
- The pipeline makes assumptions regarding the position of the vehicle. It will be good to validate these assumptions across various data sets. 
- The pipeline does not differentiate two vehicles if they are close to each other. 
- The pipeline will fail if it encounters car-like images. Road signs, bill boards, vehicle carriers, forest roads with tree cover are some scenarios.   
