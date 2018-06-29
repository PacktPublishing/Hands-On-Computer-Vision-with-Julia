ENV["PKG_CONFIG_PATH"] = "/Users/dc/anaconda/envs/python35/lib/pkgconfig"

using OpenCV
using Images, ImageView
using Cxx

# C++ OpenCV code
cxx"""

#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include <fstream>
#include <iostream>
#include <cstdlib>
#include <iomanip>
#include <string> 

using namespace std;
using namespace cv;
using namespace cv::dnn;

Net load_model(String caffe_model_txt, String caffe_model_bin) {
 
    Net net = dnn::readNetFromCaffe(caffe_model_txt, caffe_model_bin);
    return net;
}

void detect_objects(Mat img, Net net) {

    string CLASS_NAMES[] = {"background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"};

    // prepare image for evaluation
    Mat scaled_image;
    resize(img, scaled_image, Size(300,300));
    scaled_image = blobFromImage(scaled_image, 0.007843, Size(300,300), Scalar(127.5, 127.5, 127.5), false);

    // run the network
    net.setInput(scaled_image, "data");
    Mat detection_out = net.forward("detection_out");
    Mat results(detection_out.size[2], detection_out.size[3], CV_32F, detection_out.ptr<float>());
    
    // draw bounding boxes if the probability is over a specific threshold
    float threshold = 0.5;
    for (int i = 0; i < results.rows; i++) {

        float prob = results.at<float>(i, 2);

        if (prob > threshold) {
            int class_idx = static_cast<int>(results.at<float>(i, 1));
            int xLeftBottom = static_cast<int>(results.at<float>(i, 3) * img.cols);
            int yLeftBottom = static_cast<int>(results.at<float>(i, 4) * img.rows);
            int xRightTop = static_cast<int>(results.at<float>(i, 5) * img.cols);
            int yRightTop = static_cast<int>(results.at<float>(i, 6) * img.rows);

            String label = CLASS_NAMES[class_idx] + ": " + std::to_string(prob);

            Rect bounding_box((int)xLeftBottom, (int)yLeftBottom, (int)(xRightTop - xLeftBottom), (int)(yRightTop - yLeftBottom));
            rectangle(img, bounding_box, Scalar(0, 255, 0), 2);
            putText(img, label, Point(xLeftBottom, yLeftBottom - 10), FONT_HERSHEY_SIMPLEX, 0.85, Scalar(255,255,255));
        }
    }
}

"""

# We will first download and store weights in the data folder
# source: https://github.com/opencv/opencv_extra/blob/master/testdata/dnn/download_models.py
caffemodel_path = joinpath("data", "MobileNetSSD_deploy.caffemodel")
if ~isfile(caffemodel_path) download("https://drive.google.com/uc?export=download&id=0B3gersZ2cHIxRm5PMWRoTkdHdHc", caffemodel_path) end

prototxt_path = joinpath("data", "MobileNetSSD_deploy.prototxt")
if ~isfile(prototxt_path) download("https://raw.githubusercontent.com/chuanqi305/MobileNet-SSD/master/MobileNetSSD_deploy.prototxt", prototxt_path) end

# create a bounding box around face and save results in the root folder
opencv_dnn_model = @cxx load_model(pointer(prototxt_path), pointer(caffemodel_path));

filename = joinpath(pwd(), "sample-images", "cat-3352842_640.jpg");
img_opencv = imread(filename);
@cxx detect_objects(img_opencv, opencv_dnn_model);
imwrite(joinpath(pwd(), "object-detection-1.jpg"), img_opencv)

filename = joinpath(pwd(), "sample-images", "bird-3183441_640.jpg");
img_opencv = imread(filename);
@cxx detect_objects(img_opencv, opencv_dnn_model);
imwrite(joinpath(pwd(), "object-detection-2.jpg"), img_opencv)

filename = joinpath(pwd(), "sample-images", "kittens-555822_640.jpg");
img_opencv = imread(filename);
@cxx detect_objects(img_opencv, opencv_dnn_model);
imwrite(joinpath(pwd(), "object-detection-3.jpg"), img_opencv)