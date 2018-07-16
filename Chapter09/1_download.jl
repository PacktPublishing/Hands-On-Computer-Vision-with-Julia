PROJECT_IMG_FOLDER = joinpath("data", "book-covers")
DATASETS = ["test", "train"]

if ~isdir(PROJECT_IMG_FOLDER) mkdir(PROJECT_IMG_FOLDER) end

for dataset_name in DATASETS

    lines = nothing

    current_folder = joinpath(PROJECT_IMG_FOLDER, dataset_name)
    if ~isdir(current_folder) mkdir(current_folder) end

    file_name = joinpath(pwd(), "book30-listing-$dataset_name.csv")
    f = open(file_name); lines = readlines(f); close(f);
    
    for line in lines
        line_split = split(line, "\",\"")
        download(line_split[3], joinpath(PROJECT_IMG_FOLDER, dataset_name, line_split[2]))
    end
end