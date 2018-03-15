using Images

# load images from disk
sample_image_path = "sample-images/cats-3061372_640.jpg";
sample_image = nothing

if isfile(sample_image_path)
    sample_image = load(sample_image_path);
else
    info("ERROR: Image not found!")
end

# load images from url
image_url = "https://cdn.pixabay.com/photo/2018/01/04/18/58/cats-3061372_640.jpg?attachment"
downloaded_image_path = download(image_url)
downloaded_image = load(downloaded_image_path)

# download image to a predefined location
image_url = "https://cdn.pixabay.com/photo/2018/01/04/18/58/cats-3061372_640.jpg?attachment"
downloaded_image_path = download(image_url, 'sample-images/cats-3061372_640.jpg')
downloaded_image = load(downloaded_image_path)

# read directory with images
directory_path = "sample-images";
directory_files = readdir(directory_path);
directory_images = filter(x -> ismatch(r"\.(jpg|png|gif){1}$"i, x), directory_files);

for image_name in directory_images
    image_path = joinpath(directory_path, image_name);
    image = load(image_path);
    # other operations
end