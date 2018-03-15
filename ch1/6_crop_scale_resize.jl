using Images, ImageView

# crop an image
source_image = load("sample-images/cats-3061372_640.jpg");
size(source_image)
cropped_image = img[100:290, 280:540]; # (height, width)
imshow(cropped_image)

# crop an image (option 2)
cropped_image_view = view(img, 100:290, 280:540); # (height, width)
imshow(cropped_image_view)

# resize an image
source_image = load("sample-images/cats-3061372_640.jpg");
resized_image = imresize(source_image, (100, 250)); # (height, width)
imshow(resized_image);

# resize an image (2)
source_image = load("sample-images/cats-3061372_640.jpg");
resized_image = imresize(source_image, (200, 200)); # (height, width)
imshow(resized_image);