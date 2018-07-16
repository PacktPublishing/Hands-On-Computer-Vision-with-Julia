using Images, MXNet, JLD

function get_labels_from_file(path) 
  return map(x -> parse(Int8, split(x)[2]), readlines(path))
end

PROJECT_IMG_FOLDER = joinpath("data", "book-covers");
DUMP_IMG_TRAIN = joinpath(PROJECT_IMG_FOLDER, "dump-train.jld")
DUMP_IMG_TEST =  joinpath(PROJECT_IMG_FOLDER, "dump-test.jld")
LABELS_TRAIN = joinpath(PROJECT_IMG_FOLDER, "bookcover30-labels-train.txt")
LABELS_TEST = joinpath(PROJECT_IMG_FOLDER, "bookcover30-labels-test.txt")

# get the data
train_img_data = JLD.load(DUMP_IMG_TRAIN, "data");
test_img_data = JLD.load(DUMP_IMG_TEST, "data");

train_labels = get_labels_from_file(LABELS_TRAIN);
test_labels = get_labels_from_file(LABELS_TEST);

# shuffle the dataset
shuffle_train_indices = shuffle(1:size(train_labels, 1))
shuffle_test_indices = shuffle(1:size(test_labels, 1))

train_img_data = train_img_data[:, shuffle_train_indices];
train_labels = train_labels[shuffle_train_indices];

test_img_data = test_img_data[:, shuffle_test_indices];
test_labels = test_labels[shuffle_test_indices];

# create data providers
valid_record_count = 5000
train_data_provider = mx.ArrayDataProvider(:data => train_img_data[:, 1:end-valid_record_count], :label => train_labels[1:end-valid_record_count], shuffle = true, batch_size = 500);
valid_data_provider = mx.ArrayDataProvider(:data => train_img_data[:, end-valid_record_count+1:end], :label => train_labels[end-valid_record_count+1:end], shuffle = true, batch_size = 500);

arch = @mx.chain mx.Variable(:data) =>
  mx.FullyConnected(num_hidden=30) =>
  mx.SoftmaxOutput(mx.Variable(:label))

nnet = mx.FeedForward(arch, context = mx.cpu())
mx.fit(nnet, mx.ADAM(), train_data_provider, eval_data = valid_data_provider, n_epoch = 15, eval_metric = mx.MultiMetric([mx.Accuracy(), mx.ACE()]));

# testing the model
test_data_provider = mx.ArrayDataProvider(:data => test_img_data, batch_size = 500);
predictions = mx.predict(nnet, test_data_provider)
results = map(x -> findmax(predictions[:, x])[2], 1:size(test_labels, 1))

mean((results - 1) .== test_labels)