using Images, ImageView, ImageMorphology

# load an image and first convert to grayscale and then binarize
img = load("sample-images/bird-3183441_640.jpg");
img_binary = Gray.(img) .> 0.5;
img_binary = Gray.(img_binary)

# resizing images by half
img = restrict(img)
img_binary = restrict(RGB.(img_binary))

# stacking 2 images together
img_width = size(img, 2)

combined_image = fill(RGB4{Float16}(0.,0.,0.), size(img) .* (1, 2))
combined_image[:, 1:img_width] = img
combined_image[:, img_width+1:img_width*2] = img_binary
imshow(combined_image)