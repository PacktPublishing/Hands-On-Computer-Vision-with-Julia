ENV["PKG_CONFIG_PATH"] = "/Users/dc/anaconda/envs/python35/lib/pkgconfig"

using OpenCV
using Images
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

# loading an image
filename = joinpath(pwd(), "sample-images", "cat-3352842_640.jpg");
img_opencv = imread(filename);

# convert to Julia images
img_images = opencv_to_image(img_opencv);
