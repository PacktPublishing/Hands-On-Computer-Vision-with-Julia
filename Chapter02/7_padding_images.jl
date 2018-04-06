# create a border of a fixed size and fill it with a constant value
using Images, ImageView

pad_top = 25
pad_bottom = 25
pad_color = RGB4{N0f8}(0.5,0.5,0.) # border color

img = load("sample-images/cats-3061372_640.jpg");
img = padarray(img, Fill(pad_color, (pad_top, pad_bottom)))
img = parent(img) # reset indices to start from 1
imshow(img)


# create a border of a fixed size based on a content of an image
using Images, ImageView

pad_top = 25
pad_bottom = 25

img = load("sample-images/cats-3061372_640.jpg");
img = padarray(img, Pad(:reflect, pad_top, pad_bottom))
img = parent(img) # reset indices to start from 1
imshow(img)