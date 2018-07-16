using Images

img = load("sample-images/cats-3061372_640.jpg");
img_channel_view = channelview(img);
img_channel_view = permuteddimsview(img_channel_view, (2, 3, 1));
# img_channel_view[:, :, 2] *= 0.9;
img_channel_view[img_channel_view .> 0.7] *= 0.9 
imshow(img_channel_view)

img_channel_view_1d = view(img_channel_view, 1, :, :)
img_channel_view_1d = img_channel_view_1d * 2

img_float = Float16.(img_channel_view)
img_float[2, :, :] *= 0.9
colorview(RGB, img_float)
imshow(img)

img_adjusted = copy(img)
img_adjusted_channel_view = channelview(img_adjusted)
img_adjusted *= 2

img_channel_view[img_channel_view[:, :, :] .> 0.75N0f8] = img_channel_view[img_channel_view[:, :, :] .> 0.75N0f8] .+ 1N0f8
imshow(img)

img = load("sample-images/cats-3061372_640.jpg");
img_channel_view = channelview(img);
img_channel_view = permuteddimsview(img_channel_view, (2, 3, 1));

x_coords = 320:640

img_channel_view[:, x_coords, 1] = min.(img_channel_view[:, x_coords, 1] .* 1.1, 1);
img_channel_view[:, x_coords, 2] = min.(img_channel_view[:, x_coords, 2] .* 1.2, 1);
img_channel_view[:, x_coords, 3] = min.(img_channel_view[:, x_coords, 3] .* 1.4, 1);
# img_channel_view[img_channel_view .> 0.7] *= 0.9 
imshow(img)


img = load("sample-images/cats-3061372_640.jpg");
img_area_range = 320:640

img_area = img[:, img_area_range]
img_area = imfilter(img_area, Kernel.gaussian(5))
img[:, img_area_range] = img_area
imshow(img)


img = load("sample-images/cats-3061372_640.jpg");

height_range = 1:size(img, 1)
width_range = 1:size(img, 2)
noise_color = RGB4{N0f8}(0.,0.,0.)

for i = 1:2000
    img[rand(height_range, 1)[], rand(width_range, 1)[]] = noise_color
end

img = imfilter(img, Kernel.gaussian(3))
img[:, img_area_range] = img_area
imshow(img)


border_size = 50
blur_distance = 10

img = load("sample-images/cats-3061372_640.jpg");
img = padarray(img, Pad(:reflect, border_size, 0))
img = parent(img) # reset the indices

img_area_top = 1:border_size
img[img_area_top, :] = imfilter(img[img_area_top, :], Kernel.gaussian(blur_distance))

img_area_bottom = size(img, 1)-border_size:size(img, 1)
img[img_area_bottom, :] = imfilter(img[img_area_bottom, :], Kernel.gaussian(blur_distance))
imshow(img)

img_large = fill(RGB4{N0f8}(0.,0.,0.), 640, 640)
img_large[141:500, :] = img
img_large[1:140, :] = imfilter(img[1:140, :], Kernel.gaussian(10))
imshow(img_large)


colorview(RGB, Float16.(channelview(img)))
img = padarray(img, Pad(:replicate, 0, 50))
img_area_range = 1:50
img_area = img[:, img_area_range]
img_area = imfilter(img_area, Kernel.gaussian(5))
img[:, img_area_range] = img_area
imshow(img)

# // sharpen image using "unsharp mask" algorithm
# Mat blurred; double sigma = 1, threshold = 5, amount = 1;
# GaussianBlur(img, blurred, Size(), sigma, sigma);
# Mat lowContrastMask = abs(img - blurred) < threshold;
# Mat sharpened = img*(1+amount) + blurred*(-amount);
# img.copyTo(sharpened, lowContrastMask);

# parameters
gaussian_smoothing = 2
intensity = 1

# load an image and apply Gaussian filter
img = load("sample-images/cats-3061372_640.jpg");
imgb = imfilter(img, Kernel.gaussian(gaussian_smoothing));

# convert images to Float to perform mathematical operations
img_array = Float16.(channelview(img));
imgb_array = Float16.(channelview(imgb));

sharpened = img_array .* (1 + intensity) .+ imgb_array .* (-intensity);
sharpened = max.(sharpened, 0);
sharpened = min.(sharpened, 1);

img[:, 1:321] = img[:, 320:640];
img[:, 320:640] = colorview(RGB, sharpened)[:, 320:640];
imshow(img)



using Images, ImageView

pad_top = 25
pad_bottom = 25
pad_color = RGB4{N0f8}(.5,0.5,0.) # color 

img = load("sample-images/cats-3061372_640.jpg");
img = padarray(img, Fill(pad_color, (pad_top, pad_bottom)))
img = parent(img)





# load an image and create a grayscale copy
img = load("sample-images/busan-night-scene.jpg");
img_gray = RGB.(Gray.(img));

# get channel representation
img_channel_view = channelview(img);
img_gray_channel_view = channelview(img_gray);

# make channel dimension last
img_arr = permuteddimsview(img_channel_view, (2, 3, 1));
img_grayscale_arr = permuteddimsview(img_gray_channel_view, (2, 3, 1));

# create a mask with all values beiung true
img_mask = fill(true, size(img));

# spot are with colours to retain
img_spot_height = 430:460
img_spot_width = 430:460

for channel_id = 1:3

    # select current channel and crop are of interest
    current_channel = view(img_arr, :, :, channel_id)
    current_channel_area = current_channel[img_spot_height, img_spot_width, :]
    
    # identify min and max values in a cropped area
    channel_min = minimum(current_channel_area)
    channel_max = maximum(current_channel_area)

    # merge existing mask with a channel specific mask

    channel_mask = channel_min .< current_channel .< channel_max
    img_mask = img_mask .& channel_mask

    #img_mask = img_mask .& (current_channel .> channel_min) .& (current_channel .< channel_max)
end

# apply mask to original and grayscale images
img_masked = img_arr .* img_mask .+ img_grayscale_arr .* (~img_mask);
imshow(colorview(RGB, permutedims(img_masked, (3, 1, 2))))