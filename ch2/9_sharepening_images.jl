using Images, ImageView

gaussian_smoothing = 1
intensity = 1

# load an image and apply Gaussian smoothing filter
img = load("sample-images/cats-3061372_640.jpg");
imgb = imfilter(img, Kernel.gaussian(gaussian_smoothing));

# convert images to Float to perform mathematical operations
img_array = Float16.(channelview(img));
imgb_array = Float16.(channelview(imgb));

# create a sharpened version of our image and fix values from 0 to 1
sharpened = img_array .* (1 + intensity) .+ imgb_array .* (-intensity);
sharpened = max.(sharpened, 0);
sharpened = min.(sharpened, 1);
imshow(sharpened)

# optional part: comparison
img[:, 1:321] = img[:, 320:640];
img[:, 320:640] = colorview(RGB, sharpened)[:, 320:640];
imshow(img)