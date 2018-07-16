using StatsBase

PROJECT_IMG_FOLDER = joinpath("data", "book-covers")
DATASETS = ["test", "train"]

if ~isdir(PROJECT_IMG_FOLDER) mkdir(PROJECT_IMG_FOLDER) end

for dataset_name in DATASETS

    categories = Int[]

    current_folder = joinpath(PROJECT_IMG_FOLDER, dataset_name)
    file_name = joinpath(PROJECT_IMG_FOLDER, "bookcover30-labels-$dataset_name.txt")
    f = open(file_name); lines = readlines(f); close(f);
    
    for line in lines
        value = parse(Int, split(line, " ")[2])
        push!(categories, value)
    end

    println(countmap(categories))
end