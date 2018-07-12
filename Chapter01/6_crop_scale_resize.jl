using Images, ImageView

# crop an image
source_image = load("sample-images/cats-3061372_640.jpg");
size(source_image)
cropped_image = img[100:290, 280:540]; # (height, width)
imshow(cropped_image)

# crop an image (option 2)
source_image = load("sample-images/cats-3061372_640.jpg");
cropped_image_view = view(source_image, 100:290, 280:540); # (height, width)
imshow(cropped_image_view)

# resize an image
source_image = load("sample-images/cats-3061372_640.jpg");
resized_image = imresize(source_image, (100, 250)); # (height, width)
imshow(resized_image);

# resize an image (2)
source_image = load("sample-images/cats-3061372_640.jpg");
resized_image = imresize(source_image, (200, 200)); # (height, width)
imshow(resized_image);

# scale an image by percentage
source_image = load("sample-images/cats-3061372_640.jpg");
scale_percentage = 0.6
new_size = trunc.(Int, size(source_image) .* scale_percentage)
scaled_image = imresize(source_image, new_size)
imshow(resized_image);

# scale an image to a specific dimension
source_image = load("sample-images/cats-3061372_640.jpg");
new_width = 200
scale_percentage = new_width / size(source_image)[2]
new_size = trunc.(Int, size(source_image) .* scale_percentage)
scaled_image = imresize(source_image, new_size)
imshow(scaled_image);

# scale/ resize image by two-fold
source_image = load("sample-images/cats-3061372_640.jpg");
resized_image = restrict(source_image, 1); # height
imshow(resized_image);

