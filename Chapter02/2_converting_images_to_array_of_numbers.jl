using Images, ImageView
img = load("sample-images/cats-3061372_640.jpg")
img_channel_view = channelview(img)
imshow(img_channel_view)

# update all colors with a value over 0.7 to be 0.9:
using Images, ImageView
img = load("sample-images/cats-3061372_640.jpg")
img_channel_view = channelview(img)
img_channel_view[img_channel_view .> 0.7] = 0.9
imshow(img)