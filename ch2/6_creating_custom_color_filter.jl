using Images, ImageView

# load an image and create a grayscale copy
img = load("sample-images/busan-night-scene.jpg");
img_gray = RGB.(Gray.(img))

# get channels representation
img_channel_view = channelview(img);
img_gray_channel_view = channelview(img_gray);

# make channel dimension last and crop the required are
img_arr = permuteddimsview(img_channel_view, (2, 3, 1));
img_gray_arr = permuteddimsview(img_gray_channel_view, (2, 3, 1));

# create a mask with all values being true
img_mask = fill(true, size(img));

# spot are with colors to retain
img_spot_height = 430:460
img_spot_width = 430:460

# preview color area (optional)
imshow(img[img_spot_height, img_spot_width])

for channel_id = 1:3

    # select current channel and crop ares of interest
    current_channel = view(img_arr, :, :, channel_id)
    current_channel_area = current_channel[img_spot_height, img_spot_width, :]

    # identify min and max values in a cropped area
    channel_min = minimum(current_channel_area)
    channel_max = maximum(current_channel_area)

    # merge existing mask with a channel specific mask
    channel_mask = channel_min .< current_channel .< channel_max
    img_mask = img_mask .& channel_mask
end

# apply mask
img_masked = img_arr .* img_mask .+ img_gray_arr .* .~(img_mask);
final_image = colorview(RGB, permutedims(img_masked, (3, 1, 2)))
imshow(final_image)