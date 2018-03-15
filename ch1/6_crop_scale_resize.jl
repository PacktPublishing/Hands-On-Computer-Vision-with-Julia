using Images, ImageView

# crop an image
source_image = load("sample-images/cats-3061372_640.jpg");
size(source_image)
cropped_image = img[100:290, 280:540]; # (height, width)
imshow(cropped_image)

# crop an image (option 2)
cropped_image_view = view(img, 100:290, 280:540); # (height, width)
imshow(cropped_image_view)