using Images, ImageView, ImageSegmentation

# convert image segments to mean color value
segment_to_image(segments) = map(i->segment_mean(segments, i), labels_map(segments))

# one object
img = load("sample-images/cards-2946773_640.jpg");
#imshow(img);


bw = Gray.(img).>0.5
Gray.(bw)
dist = 1.-distance_transform(feature_transform(bw));
Gray.(dist)
markers = label_components(dist.<-5);
segments = watershed(dist, markers);
Gray.(segment_to_image(segments))