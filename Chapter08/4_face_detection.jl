ENV["PKG_CONFIG_PATH"] = "/Users/dc/anaconda/envs/python35/lib/pkgconfig"

using OpenCV
using Images, ImageView
using Cxx

# C++ OpenCV code
cxx"""

#include "opencv2/objdetect.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

#include <iostream>
#include <stdio.h>

using namespace std;
using namespace cv;

/// Load cascade file
CascadeClassifier load_face_cascade(String face_cascade_name) {

    CascadeClassifier face_cascade;
    face_cascade.load(face_cascade_name);

    return face_cascade;
}

/// detect and draw a bounding box on the image
void detect_face(cv::Mat frame, CascadeClassifier face_cascade) {
   
    std::vector<Rect> faces;
    cv::Mat frame_gray;
    std::vector<int>vec(4);

    cvtColor( frame, frame_gray, COLOR_BGR2GRAY );
    equalizeHist( frame_gray, frame_gray );

    face_cascade.detectMultiScale( frame_gray, faces, 1.1, 2, 0|CASCADE_SCALE_IMAGE, Size(30, 30) );

    for( size_t i = 0; i < faces.size(); i++ ) {
        Point center( faces[i].x + faces[i].width/2, faces[i].y + faces[i].height/2 );
        ellipse( frame, center, Size( faces[i].width/2, faces[i].height/2), 0, 0, 360, Scalar( 255, 0, 255 ), 4, 8, 0 );
    }    
}

/// detect and return bounding box coordinates
std::vector<int> detect_face_coords(cv::Mat frame, CascadeClassifier face_cascade) {
   
    std::vector<Rect> faces;
    cv::Mat frame_gray;
    std::vector<int>vec(4);

    cvtColor( frame, frame_gray, COLOR_BGR2GRAY );
    equalizeHist( frame_gray, frame_gray );

    face_cascade.detectMultiScale( frame_gray, faces, 1.1, 2, 0|CASCADE_SCALE_IMAGE, Size(30, 30) );

    if (faces.size() > 0) {
        vec[0] = faces[0].x; 
        vec[1] = faces[0].width;
        vec[2] = faces[0].y;
        vec[3] = faces[0].height;
    }
    else {
        vec[0] = 0; vec[1] = 0;
        vec[2] = 0; vec[3] = 0;
    }
    
    return vec;
}

"""

cascade_path = download("https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_alt.xml")
face_cascade = @cxx load_face_cascade(pointer(cascade_path));

# create a bounding box around face and save results in the root folder
filename = joinpath(pwd(), "sample-images", "beautiful-1274051_640_100_1.jpg");
img_opencv = imread(filename);
@cxx detect_face(img_opencv, face_cascade);
imwrite(joinpath(pwd(), "result.jpg"), img_opencv)

# retrieve coordinates and crop the image
filename = joinpath(pwd(), "sample-images", "beautiful-1274051_640_100_1.jpg");
img_opencv = imread(filename);
img_images = load(filename);

coords_cxx = @cxx detect_face_coords(img_opencv, face_cascade);
coords = map(x -> Int(at(coords, x)), 0:3);

face = img_images[coords[3]:coords[3] + coords[4], coords[1]:coords[1] + coords[2]]

