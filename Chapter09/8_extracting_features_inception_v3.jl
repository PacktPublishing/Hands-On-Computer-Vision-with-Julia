using Images, MXNet, JLD

nnet = mx.load_checkpoint("weights/inception-v3/InceptionV3-FE", 0, mx.FeedForward; context = mx.gpu());

PROJECT_IMG_FOLDER = joinpath("data", "book-covers");
DATASETS = ["train", "test"];
BATCH_SIZE = 100;

for dataset_name in DATASETS

    categories = Int[]

    current_folder = joinpath(PROJECT_IMG_FOLDER, dataset_name)
    file_name = joinpath(PROJECT_IMG_FOLDER, "bookcover30-labels-$dataset_name.txt")
    lines = readlines(file_name);
    
    results = zeros(Float16, (2048, size(lines, 1)))

    for i=1:BATCH_SIZE:size(lines, 1)
        
        println("BATCH: $i")

        mx_data = mx.zeros((299, 299, 3, BATCH_SIZE));

        for j=1:BATCH_SIZE

            if j % 50 == 0
                println("\tIMG: $j")
            end

            img = imresize(load(joinpath(current_folder, split(lines[i + j - 1])[1])), (299, 299));
            img = permutedims(Float16.(channelview(RGB.(img))), (3, 2, 1));

            img[:, :, :] *= 256.0;
            img[:, :, :] -= 128.;
            img[:, :, :] /= 128.;

            img = reshape(img, 299, 299, 3, 1);
            mx_data[j:j] = img;
        end

        data_provider = mx.ArrayDataProvider(:data => mx_data);
        results[:, i:i+(BATCH_SIZE-1)] = mx.predict(nnet, data_provider)
    end

    JLD.save(joinpath(PROJECT_IMG_FOLDER, "dump-$dataset_name-inception.jld"), "data", results)
end

