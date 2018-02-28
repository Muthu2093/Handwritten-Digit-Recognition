%% Function to train and test the MNIST data set using the autoencoder demo code
%% EXTRA CREDIT
%% Changed to train and test only 10,000 and 2,000 images respectively
%% Change Line 13,14 for changed training set size and Line 72,73 for changing test set size
%% Highest the data set size, more efficiency

clc;
close all;
clear all;

%% Reading the training images
load('mnist.mat');
xTrainImages=train_Images(:,1:10000);
tTrain=train_labels(:,1:10000);

% Display some of the training images
clf
for i = 1:20
    subplot(4,5,i);
    imshow(xTrainImages{i});
end

rng('default');
hiddenSize1 = 100;
autoenc1 = trainAutoencoder(xTrainImages,hiddenSize1, ...
    'MaxEpochs',400, ...
    'L2WeightRegularization',0.004, ...
    'SparsityRegularization',4, ...
    'SparsityProportion',0.15, ...
    'ScaleData', false);
view(autoenc1);

%% Visualizaing weights of encoders

figure()
plotWeights(autoenc1);

feat1 = encode(autoenc1,xTrainImages);

%% Training second autoencoder

hiddenSize2 = 50;
autoenc2 = trainAutoencoder(feat1,hiddenSize2, ...
    'MaxEpochs',100, ...
    'L2WeightRegularization',0.002, ...
    'SparsityRegularization',4, ...
    'SparsityProportion',0.1, ...
    'ScaleData', false);

view(autoenc2)

feat2 = encode(autoenc2,feat1);


%% Training final softmax layer
softnet = trainSoftmaxLayer(feat2,tTrain,'MaxEpochs',400);
view(softnet)


%% Forming a stacked neural network

deepnet = stack(autoenc1,autoenc2,softnet);
view(deepnet)

% Get the number of pixels in each image
imageWidth = 20;
imageHeight = 20;
inputSize = imageWidth*imageHeight;

%% Load the test images
load('mnistTest.mat');
xTestImages=test_Images(:,1:2000);
tTest=test_labels(:,1:2000);

% Turn the test images into vectors and put them in a matrix
xTest = zeros(inputSize,numel(xTestImages));
for i = 1:numel(xTestImages)
    xTest(:,i) = xTestImages{i}(:);
end

y = deepnet(xTest);
plotconfusion(tTest,y);

% Turn the training images into vectors and put them in a matrix
xTrain = zeros(inputSize,numel(xTrainImages));
for i = 1:numel(xTrainImages)
    xTrain(:,i) = xTrainImages{i}(:);
end

%% Perform fine tuning
deepnet = train(deepnet,xTrain,tTrain);

y = deepnet(xTest);
plotconfusion(tTest,y);

