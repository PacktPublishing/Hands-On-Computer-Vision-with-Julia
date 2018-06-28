ENV["PKG_CONFIG_PATH"] = "/Users/dc/anaconda/envs/python35/lib/pkgconfig"

using OpenCV
using Images, ImageView
using Cxx

function opencv_to_image(img_opencv)

    converted_image = zeros(Float16, (3, rows(img_opencv), cols(img_opencv)));

    for i = 1:size(converted_image, 2)
        for j = 1:size(converted_image, 3)
            pixel_value = @cxx at_v3b(img_opencv, i, j)
            converted_image[:, i, j] = map(x -> Int(at(pixel_value, x)), [2, 1, 0]) ./ 255
        end
    end

    return converted_image
end

# C++ OpenCV code
cxx"""

#include "opencv2/videoio.hpp"
#include "opencv2/imgproc.hpp"

using namespace std;
using namespace cv;

/// Retrieve video device by index
cv::VideoCapture get_video_device(int device_index) {

    cv::VideoCapture capture(device_index);
    cv::Mat frame;

    capture.read(frame);

    return capture;
}

/// Capture frame
cv::Mat capture_frame(cv::VideoCapture capture) {
   
    cv::Mat frame;
    bool Success = capture.read(frame);
    return frame;
}

/// Capture and save frame
void capture_save_frame(cv::VideoCapture capture, String dest) {
   
    cv::Mat frame;
    bool success = capture.read(frame);

    if (success) {
        cv::imwrite(dest, frame);
    }
}


/// Release an active camera
void release_camera(cv::VideoCapture capture) {
    capture.release();
}

"""

video_device = @cxx get_video_device(CAP_ANY);

@time current_frame = @cxx capture_frame(video_device);
@time current_frame_image = opencv_to_image(current_frame);
Images.imshow(colorview(RGB, current_frame_image))

filename = joinpath(pwd(), "camera-frame.jpg");
@time @cxx capture_save_frame(video_device, pointer(filename));
@time current_frame_image_2 = load(filename);
imshow(current_frame_image_2)