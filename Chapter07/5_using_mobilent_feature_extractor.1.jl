using Images, MXNet, MLDatasets, StatsBase

DATASET_DIR = "data/101_ObjectCategories"
CLASSES = readdir(DATASET_DIR)

images = Tuple{String,Int}[]

for dir_id = 1:length(CLASSES)

    dir = CLASSES[dir_id]
    all_files = readdir(joinpath(DATASET_DIR, dir))
    keep_files = filter(x -> endswith(x, ".jpg"), all_files)

    append!(images, map(x -> (joinpath(DATASET_DIR, dir, x), dir_id), keep_files))
end

### LOAD THE NEURAL NETWORK
nnet = mx.load_checkpoint("weights/inception-v3/InceptionV3-FE", 0, mx.FeedForward; context = mx.gpu());
nnet.arch = @mx.chain nnet.arch => mx.FullyConnected(num_hidden=length(CLASSES)) => mx.SoftmaxOutput(mx.Variable(:label))

### SHUFFLE DATASET
images = images[shuffle(1:length(images))];

### DEFINE SIZE OF TRAIN AND TEST DATASETS
train_indices = 1:8000
valid_indices = 8000:length(images)
batch_size = length(valid_indices)

optimizer = mx.ADAM()

for i = 1:100

    println(i, " - ", i+batch_size-1);

    train_indices_batch = sample(train_indices, batch_size, replace = false);
    valid_indices_batch = sample(valid_indices, batch_size, replace = false);

    mx_data_train = mx.zeros((299, 299, 3, batch_size));
    mx_data_train_y = zeros(Int, batch_size);

    mx_data_valid = mx.zeros((299, 299, 3, batch_size));
    mx_data_valid_y = zeros(Int, batch_size);

    for idx = 1:batch_size

        mx_data_train[idx:idx] = reshape(Float16.(channelview(RGB.(imresize(load(images[train_indices_batch[idx]][1]), (299, 299))))), (299, 299, 3, 1));
        mx_data_train_y[idx] = images[train_indices_batch[idx]][2];

        mx_data_valid[idx:idx] = reshape(Float16.(channelview(RGB.(imresize(load(images[valid_indices_batch[idx]][1]), (299, 299))))), (299, 299, 3, 1));
        mx_data_valid_y[idx] = images[valid_indices_batch[idx]][2];
    end

    mx_data_train *= 256.0;
    mx_data_train -= 128.;
    mx_data_train /= 128.;

    mx_data_valid *= 256.0;
    mx_data_valid -= 128.;
    mx_data_valid /= 128.;

    train_data_provider = mx.ArrayDataProvider(:data => mx_data_train, :label => mx_data_train_y, batch_size = 25);
    validation_data_provider = mx.ArrayDataProvider(:data => mx_data_valid, :label => mx_data_valid_y, batch_size = 25);

    mx.fit(nnet, optimizer, train_data_provider, eval_data = validation_data_provider, n_epoch = 3, callbacks = [mx.speedometer()]);

end


### PREPARING THE INPUT
batch_size = 100
train_indices = 1:40000
validation_indices = 40001:50000
optimizer = mx.ADAM();

for i = 1:10

    println(i, " - ", i+batch_size-1)

    train_indices_batch = sample(train_indices, batch_size, replace = false)
    validation_indices_batch = sample(validation_indices, batch_size, replace = false)

    mx_data_train = mx.zeros((299, 299, 3, batch_size));
    mx_data_validation = mx.zeros((299, 299, 3, batch_size));

    map(x -> mx_data_train[x:x] = reshape(Float16.(channelview(imresize(train_x[:, :, :, train_indices_batch[x]], (299, 299)))), (299, 299, 3, 1)), 1:batch_size);
    map(x -> mx_data_validation[x:x] = reshape(Float16.(channelview(imresize(train_x[:, :, :, validation_indices_batch[x]], (299, 299)))), (299, 299, 3, 1)), 1:batch_size);
    
    mx_data_train *= 256.0;
    mx_data_train -= 128.;
    mx_data_train /= 128.;

    mx_data_validation *= 256.0;
    mx_data_validation -= 128.;
    mx_data_validation /= 128.;

    train_data_provider = mx.ArrayDataProvider(:data => mx_data_train, :label => train_y[train_indices_batch], batch_size = 50);
    validation_data_provider = mx.ArrayDataProvider(:data => mx_data_validation, :label => train_y[validation_indices_batch], batch_size = 100);

    mx.fit(nnet, optimizer, train_data_provider, eval_data = validation_data_provider, n_epoch = 5, callbacks = [mx.speedometer()]);
end

features_train = features[:, 1:4000]
features_validate = features[:, 4001:end];
features_train_y = outputs[1:4000]
features_validate_y = outputs[4001:end]

train_length = 4000
validation_length = 1000

train_data_array  = mx.zeros((size(features_train, 1)..., train_length...));
train_label_array = mx.zeros(train_length);

validation_data_array  = mx.zeros((size(features_train, 1)..., validation_length...));
validation_label_array = mx.zeros(validation_length);

# The number of records we send to the training should be at least number of outcome
for idx = 1:train_length
    train_data_array[idx:idx] = reshape(features_train[:, idx], (size(features_train, 1)..., 1...))
    train_label_array[idx:idx] = features_train_y[idx]
end

for idx = 1:validation_length
    validation_data_array[idx:idx] = reshape(features_validate[:, idx], (size(features_validate, 1)..., 1...))
    validation_label_array[idx:idx] = features_validate_y[idx]
end

train_data_provider = mx.ArrayDataProvider(:data => features_train, :label => train_label_array, batch_size = 500);
validation_data_provider = mx.ArrayDataProvider(:data => features_validate, :label => validation_label_array, batch_size = 500);


arch = @mx.chain mx.Variable(:data) =>
  mx.FullyConnected(num_hidden=128) =>
  mx.Activation(act_type=:relu) =>
  mx.FullyConnected(num_hidden=64) =>
  mx.Activation(act_type=:relu) =>
  mx.FullyConnected(num_hidden=10) =>
  mx.SoftmaxOutput(mx.Variable(:label))


arch = @mx.chain mx.Variable(:data) =>
  mx.FullyConnected(num_hidden=128) =>
  mx.Activation(act_type=:relu) =>
  mx.FullyConnected(num_hidden=64) =>
  mx.Activation(act_type=:relu) =>
  mx.FullyConnected(num_hidden=10) =>
  mx.Activation(act_type=:relu) =>
  mx.SoftmaxOutput(mx.Variable(:label))

nnet = mx.FeedForward(arch, context = mx.gpu())

mx.fit(nnet, mx.ADAM(), train_data_provider, eval_data = validation_data_provider, n_epoch = 250, callbacks = [mx.speedometer()]);
mx.fit(nnet, mx.ADAM(), train_data_provider, eval_data = train_data_provider, n_epoch = 250, callbacks = [mx.speedometer()]);