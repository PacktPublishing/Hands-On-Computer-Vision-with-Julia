# changing color saturation by channel
using Images, ImageView
img = load("sample-images/cats-3061372_640.jpg");
img_ch_view = channelview(img);
img_ch_view = permuteddimsview(img_ch_view, (2, 3, 1));

x_coords = 320:640

img_ch_view[:, x_coords, 1] = min.(img_ch_view[:, x_coords, 1] .* 1.1, 1);
img_ch_view[:, x_coords, 2] = min.(img_ch_view[:, x_coords, 2] .* 1.2, 1);
img_ch_view[:, x_coords, 3] = min.(img_ch_view[:, x_coords, 3] .* 1.4, 1);
imshow(img) 