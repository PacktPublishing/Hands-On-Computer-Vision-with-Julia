using Images, ImageView

PROJECT_IMG_FOLDER = joinpath("data", "book-covers")
DATASETS = ["test", "train"]

preview_count = 5
shape = (100, 78)
preview_img = zeros(typeof(RGB4{N0f8}(0.,0.,0.)), (100, 1));

original_shapes = []

for dataset_name in DATASETS

    current_folder = joinpath(PROJECT_IMG_FOLDER, dataset_name)
    files = filter(x -> contains(x, ".jpg"), readdir(current_folder))

    shuffle!(files)

    for i=1:preview_count
        
        img_original = load(joinpath(current_folder, files[i]))
        img_resized = imresize(img_original, shape)
        
        preview_img = hcat(preview_img, img_resized)
        push!(original_shapes, size(img_original))
    end
end

println(original_shapes)