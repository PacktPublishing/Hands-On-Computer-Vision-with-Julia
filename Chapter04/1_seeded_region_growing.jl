using Images, ImageView, ImageSegmentation

# one object
img = load("sample-images/cat-3352842_640.jpg");
imshow(img);

seeds = [
    (CartesianIndex(220,250), 1), # object location
    (CartesianIndex(220,500), 2)  # background location
]

segments = seeded_region_growing(img, seeds)

imshow(map(i->segment_mean(segments,i), labels_map(segments)))

# multiple objects
img = load("sample-images/kittens-555822_640.jpg");
imshow(img);

seeds = [
    (CartesianIndex(130,90), 1), 
    (CartesianIndex(130,200), 2), 
    (CartesianIndex(130,300), 2), 
    (CartesianIndex(130,420), 3), 
    (CartesianIndex(30,50), 4)
]

segments = seeded_region_growing(img, seeds)

imshow(map(i->segment_mean(segments,i), labels_map(segments)))

# multiple objects
img = load("sample-images/bird-3183441_640.jpg");

seeds = [
    (CartesianIndex(240,120), 1), 
    (CartesianIndex(295,70), 2),
    (CartesianIndex(319,40), 3),
    (CartesianIndex(90,300), 4),
    (CartesianIndex(295,325), 5),
    (CartesianIndex(76,135), 6)
]

segments = seeded_region_growing(img, seeds)

imshow(map(i->segment_mean(segments,i), labels_map(segments)))

