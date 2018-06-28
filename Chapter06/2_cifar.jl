using Images, MLDatasets, MXNet#, ImageView

train_x, train_y = CIFAR10.traindata();
test_x,  test_y  = CIFAR10.testdata();

# preview the images from train_x dataset
preview_img = zeros((0..., size(train_x, 1, 3)...));

for i = 1:10
    preview_img = vcat(preview_img, train_x[:, :, :, i])
end

# colorview(RGB, permutedims(preview_img, (3, 2, 1)))

imshow(colorview(RGB, permutedims(preview_img, (3, 2, 1))))

# creating first neural network
using MXNet

train_length = 40000
validation_length = 10000

train_data_array  = mx.zeros((size(train_x, 1, 2, 3)..., train_length...));
train_label_array = mx.zeros(train_length);

validation_data_array  = mx.zeros((size(train_x, 1, 2, 3)..., validation_length...));
validation_label_array = mx.zeros(validation_length);

test_data_array  = mx.zeros((size(train_x, 1, 2, 3)..., size(test_x, 4)...));
test_label_array = mx.zeros(size(test_x, 4));

# The number of records we send to the training should be at least number of outcome
for idx = 1:train_length
    train_data_array[idx:idx] = reshape(train_x[:, :, :, idx], (size(train_x, 1, 2,3 )..., 1...))
    train_label_array[idx:idx] = train_y[idx]
end

for idx = 1:validation_length
    validation_data_array[idx:idx] = reshape(train_x[:, :, :, train_length + idx], (size(train_x, 1, 2,3 )..., 1...))
    validation_label_array[idx:idx] = train_y[train_length + idx]
end

for idx = 1:size(test_x, 4)
    test_data_array[idx:idx] = reshape(test_x[:, :, :, idx], (size(test_x, 1, 2,3 )..., 1...))
    test_label_array[idx:idx] = test_y[idx]
end

train_data_provider = mx.ArrayDataProvider(:data => train_data_array, :label => train_label_array, batch_size = 500);
validation_data_provider = mx.ArrayDataProvider(:data => validation_data_array, :label => validation_label_array, batch_size = 500);
test_data_provider = mx.ArrayDataProvider(:data => validation_data_array, :label => validation_label_array, batch_size = 500);

# more complicated network
arch = @mx.chain mx.Variable(:data) =>
  mx.FullyConnected(num_hidden=128) =>
  mx.Activation(act_type=:relu) =>
  mx.FullyConnected(num_hidden=64) =>
  mx.Activation(act_type=:relu) =>
  mx.FullyConnected(num_hidden=10) =>
  mx.Activation(act_type=:relu) =>
  mx.SoftmaxOutput(mx.Variable(:label))

nnet = mx.FeedForward(arch, context = mx.cpu())

mx.fit(nnet, mx.ADAM(), train_data_provider, eval_data = validation_data_provider, n_epoch = 250, callbacks = [mx.speedometer()]);

### UPDATED NETWORK 1

arch = @mx.chain mx.Variable(:data) =>
        mx.Convolution(kernel=(8, 8), num_filter=16, stride = (4, 4)) =>
        mx.Activation(act_type=:relu) =>
        mx.Convolution(kernel=(4, 4), num_filter=32, stride = (2, 2)) =>
        mx.Activation(act_type=:relu) =>
        mx.Pooling( kernel=(2, 2), pool_type=:max) =>
        mx.Flatten() =>
        mx.FullyConnected(num_hidden=256) =>
        mx.Activation(act_type=:relu) =>
        mx.FullyConnected(num_hidden=10) =>
        mx.SoftmaxOutput(mx.Variable(:label))

nnet = mx.FeedForward(arch, context = mx.cpu())
mx.fit(nnet, mx.ADAM(), train_data_provider, eval_data = validation_data_provider, n_epoch = 250, callbacks = [mx.speedometer()]);

###

model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(3, 32, 32), padding='same', activation='relu', kernel_constraint=maxnorm(3)))
model.add(Dropout(0.2))
model.add(Conv2D(32, (3, 3), activation='relu', padding='same', kernel_constraint=maxnorm(3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(512, activation='relu', kernel_constraint=maxnorm(3)))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))
# Compile model
epochs = 25
lrate = 0.01
decay = lrate/epochs
sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
print(model.summary())


arch = @mx.chain mx.Variable(:data) =>

  mx.Convolution(kernel=(3, 3), num_filter=32) =>
  mx.Activation(act_type=:relu) =>
  mx.Dropout(p = 0.2) =>
  mx.Convolution(kernel=(3, 3), num_filter=32) =>
  mx.Activation(act_type=:relu) =>
  mx.Pooling(kernel=(2, 2), pool_type=:max) =>
  mx.Flatten() => 
  mx.FullyConnected(num_hidden=512) => 
  mx.Dropout(p = 0.5) =>
  mx.FullyConnected(num_hidden=10) =>
  mx.SoftmaxOutput(mx.Variable(:label))


arch = @mx.chain mx.Variable(:data) =>
        mx.Convolution(kernel=(3, 3), num_filter=32) =>
        mx.Activation(act_type=:relu) =>
        mx.Dropout(p = 0.25) => 
        mx.Pooling( kernel=(2, 2), pool_type=:max) =>
        mx.Flatten() =>
        mx.FullyConnected(num_hidden=512) =>
        mx.Activation(act_type=:relu) =>
        mx.FullyConnected(num_hidden=10) =>
        mx.SoftmaxOutput(mx.Variable(:label))

nnet = mx.FeedForward(arch, context = mx.cpu())
mx.fit(nnet, mx.SGD(momentum = 0.9, weight_decay = 0.01/25), train_data_provider, eval_data = validation_data_provider, n_epoch = 250);
        

arch = @mx.chain mx.Variable(:data) =>
        mx.Convolution(kernel=(7, 7), num_filter=32) =>
        mx.Activation(act_type=:relu) =>
        mx.Pooling( kernel=(2, 2), pool_type=:max) =>
        mx.Flatten() =>
        mx.FullyConnected(num_hidden=512) =>
        mx.Activation(act_type=:relu) =>
        mx.FullyConnected(num_hidden=10) =>
        mx.SoftmaxOutput(mx.Variable(:label))


builder.Convolution((7, 7), 32),
        builder.ReLU(),
        builder.Pooling('max', (2, 2), (2, 2)),
        builder.Flatten(),
        builder.Affine(hidden_size),
        builder.Affine(num_classes),



# UPDATED NETWORK 2
model.add(Conv2D(32, (3, 3), padding='same',
                 input_shape=x_train.shape[1:]))
model.add(Activation('relu'))
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes))
model.add(Activation('softmax'))

arch = @mx.chain mx.Variable(:data) =>

  mx.Convolution(kernel=(3, 3), num_filter=32) =>
  mx.Activation(act_type=:relu) =>
  mx.Convolution(kernel=(3, 3), num_filter=32) =>
  mx.Activation(act_type=:relu) =>
  mx.Pooling( kernel=(2, 2), pool_type=:max) =>
  mx.Dropout(p = 0.25) =>

  mx.Convolution(kernel=(3, 3), num_filter=64) =>
  mx.Activation(act_type=:relu) =>
  mx.Convolution(kernel=(3, 3), num_filter=64) =>
  mx.Activation(act_type=:relu) =>
  mx.Pooling( kernel=(2, 2), pool_type=:max) =>
  mx.Dropout(p = 0.25) =>

  mx.Flatten() => 
  mx.FullyConnected(num_hidden=256) => 
  mx.Dropout(p = 0.5) =>

  mx.FullyConnected(num_hidden=10) =>
  mx.SoftmaxOutput(mx.Variable(:label))


arch = @mx.chain mx.Variable(:data) =>

  mx.Convolution(kernel=(3, 3), num_filter=64, stride = (1, 1)) =>
  mx.Activation(act_type=:relu) =>
  mx.Convolution(kernel=(3, 3), num_filter=64, stride = (1, 1)) =>
  mx.Activation(act_type=:relu) =>
  mx.Pooling( kernel=(2, 2), stride=(2, 2), pool_type=:max) =>

  mx.Convolution(kernel=(3, 3), num_filter=128, stride = (1, 1)) =>
  mx.Activation(act_type=:relu) =>
  mx.Convolution(kernel=(3, 3), num_filter=64, stride = (1, 1)) =>
  mx.Activation(act_type=:relu) =>
  mx.Pooling( kernel=(2, 2), stride=(2, 2), pool_type=:max) =>

  mx.Convolution(kernel=(3, 3), num_filter=128, stride = (1, 1)) =>
  mx.Activation(act_type=:relu) =>
  mx.Pooling( kernel=(2, 2), stride=(2, 2), pool_type=:max) =>

  mx.Dropout(p = 0.25) =>
  mx.Flatten() => 
  mx.FullyConnected(num_hidden=512) => 
  mx.Dropout(p = 0.5) =>

  mx.FullyConnected(num_hidden=10) =>
  mx.SoftmaxOutput(mx.Variable(:label))

# arch = @mx.chain mx.Variable(:data) =>

#   mx.Convolution(kernel=(3, 3), num_filter=64, stride = (1, 1)) =>
#   mx.Activation(act_type=:relu) =>
#   mx.Pooling( kernel=(2, 2), stride=(2, 2), pool_type=:max) =>

#   mx.Convolution(kernel=(3, 3), num_filter=128, stride = (1, 1)) =>
#   mx.Activation(act_type=:relu) =>
#   mx.Pooling( kernel=(2, 2), stride=(2, 2), pool_type=:max) =>

#   mx.Convolution(kernel=(3, 3), num_filter=256, stride = (1, 1)) =>
#   mx.Activation(act_type=:relu) =>
#   mx.Pooling( kernel=(2, 2), stride=(2, 2), pool_type=:max) =>

#   mx.Dropout(p = 0.5) =>
#   mx.Flatten() => 
#   mx.FullyConnected(num_hidden=1024) => 
#   mx.Dropout(p = 0.5) =>

#   mx.FullyConnected(num_hidden=10) =>
#   mx.SoftmaxOutput(mx.Variable(:label))

nnet = mx.FeedForward(arch, context = mx.cpu())
mx.fit(nnet, mx.ADAM(), train_data_provider, eval_data = validation_data_provider, n_epoch = 250);
mx.fit(nnet, mx.ADAM(), train_data_provider, eval_data = validation_data_provider, n_epoch = 250, callbacks = [mx.speedometer()]);
