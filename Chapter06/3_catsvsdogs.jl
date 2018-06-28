using Images, MXNet #,ImageView

# start by loading a single image
IMAGES_PATH = "data/train"

single_img = load(joinpath(IMAGES_PATH, "cat.1.jpg"))
size(single_img)

preview_img = zeros(100, 0);

for i = 1:10
    seq_x_img = vcat(
        imresize(load(joinpath($IMAGES_PATH, "cat.$i.jpg")), (50, 50)),
        imresize(load(joinpath($IMAGES_PATH, "dog.$i.jpg")), (50, 50))
    )
    preview_img = hcat(preview_img, seq_x_img)
end

imshow(preview_img)

### PREPARING THE DATASET

files = readdir(IMAGES_PATH);
data_x = zeros((32, 32, 3, size(files, 1)));
data_y = zeros(size(files, 1));

for idx = 1:size(files, 1)
    file_name = joinpath(IMAGES_PATH, files[idx])

    if endswith(file_name, ".jpg")
        img = imresize(load(file_name), (32, 32))
        try
            data_x[:, :, :, idx] = permuteddimsview(Float16.(channelview(img)), (2, 3, 1))
            data_y[idx] = 1 * contains(files[idx], "dog")
        catch
            data_x[:, :, :, idx] = permuteddimsview(Float16.(channelview(RGB.(img))), (2, 3, 1))
            data_y[idx] = 1 * contains(files[idx], "dog")
        end
    end
end

total_count = size(data_y, 1);
indices = shuffle(1:total_count);

total_train_count = Int(total_count * 0.8);
total_validation_count = total_count - total_train_count

train_data_array  = mx.zeros((size(data_x, 1, 2, 3)..., total_train_count...));
train_label_array = mx.zeros(total_train_count);

validation_data_array  = mx.zeros((size(data_x, 1, 2, 3)..., total_validation_count...));
validation_label_array = mx.zeros(total_validation_count);

for idx = 1:total_train_count
    train_data_array[idx:idx] = reshape(data_x[:, :, :, indices[idx]], (size(data_x, 1, 2, 3 )..., 1...))
    train_label_array[idx:idx] = data_y[indices[idx]]
end

for idx = 1:total_validation_count
    validation_data_array[idx:idx] = reshape(data_x[:, :, :, indices[total_train_count + idx]], (size(data_x, 1, 2,3 )..., 1...))
    validation_label_array[idx:idx] = data_y[indices[total_train_count + idx]]
end

train_data_provider = mx.ArrayDataProvider(:data => train_data_array, :label => train_label_array, batch_size = 100, shuffle = true);
validation_data_provider = mx.ArrayDataProvider(:data => validation_data_array, :label => validation_label_array, batch_size = 100, shuffle = true);

arch = @mx.chain mx.Variable(:data) =>
        mx.Convolution(kernel=(3, 3), num_filter=32) =>
        mx.Activation(act_type=:relu) =>
        mx.Dropout(p = 0.25) => 
        mx.Pooling( kernel=(2, 2), pool_type=:max) =>
        mx.Flatten() =>
        mx.FullyConnected(num_hidden=256) =>
        mx.Activation(act_type=:relu) =>
        mx.FullyConnected(num_hidden=10) =>
        mx.SoftmaxOutput(mx.Variable(:label))

nnet = mx.FeedForward(arch, context = mx.cpu())
mx.fit(nnet, mx.ADAM(), train_data_provider, eval_data = validation_data_provider, n_epoch = 50, initializer = mx.XavierInitializer());

### Improved neural network

arch = @mx.chain mx.Variable(:data) =>
        mx.Convolution(kernel=(3, 3), num_filter=32) =>
        mx.Activation(act_type=:relu) =>
        mx.Convolution(kernel=(3, 3), num_filter=32) =>
        mx.Activation(act_type=:relu) =>
        mx.Dropout(p = 0.25) => 
        mx.Pooling( kernel=(3, 3), pool_type=:max) =>
        mx.Flatten() =>
        mx.FullyConnected(num_hidden=256) =>
        mx.Activation(act_type=:relu) =>
        mx.FullyConnected(num_hidden=10) =>
        mx.SoftmaxOutput(mx.Variable(:label))

nnet = mx.FeedForward(arch, context = mx.cpu())
mx.fit(nnet, mx.ADAM(), train_data_provider, eval_data = validation_data_provider, n_epoch = 50, initializer = mx.XavierInitializer());