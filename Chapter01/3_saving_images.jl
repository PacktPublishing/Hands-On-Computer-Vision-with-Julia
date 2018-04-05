using Images

img = load("sample-images/cats-3061372_640.jpg")
save("cats-small-file-size.jpg", img) # save file in JPG format
save("cats-high-quality.png", img) # save file in PNG format

