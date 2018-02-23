%
%   TAE - Final Project
%
%   Giovanni Ballarin, Stefanie Bertele
%

clear;
close all;

% Import MNIST dataset:

X_mnist = csvread('digits_data.csv');
y_mnist = csvread('digits_labels.csv');

n_mnist = size(X_mnist, 1);
p_mnist = size(X_mnist, 2);

% Import Fashion MNIST dataset:

X_fashion = transpose(loadMNISTImages('fashion-mnist-images-idx3-ubyte'));
y_fashion = loadMNISTLabels('fashion-mnist-labels-idx1-ubyte');

X_fashion = X_fashion(1:5000, :);
y_fashion = y_fashion(1:5000, :);

n_fashion = size(X_fashion, 1);
p_fashion = size(X_fashion, 2);
 
% We are using display_network from the autoencoder code
% displayData(X_fashion(1:100, :));
% disp(y_fashion(1:10));

num_labels = 10;


Y_mnist = zeros(n_mnist, num_labels);
for j = 1:n_mnist
    Y_mnist(j, y_mnist(j)) = 1;
end

Y_fashion = zeros(n_fashion, num_labels);
for j = 1:n_fashion
    Y_fashion(j, y_fashion(j)+1) = 1;
end

%% PCA + ANN:

NN_hidden_size  = 25;
PCA_sizes       = [10, 25, 50, 100];

% MNIST

NN_PCA_mnist_error = zeros(length(PCA_sizes),1);
j = 1;

for PC_dim = PCA_sizes

    % PCA of the MNIST data:
    X_mnist_pca = X_mnist * pca(X_mnist);
    X_mnist_pca = X_mnist_pca(:, 1:PC_dim);

    % Neural network:
    net = feedforwardnet([NN_hidden_size, 20, 15], 'trainrp');
    net = train(net, X_mnist_pca',  Y_mnist', [], [], [], 'useParallel');
    [~, NN_class] = max(net(X_mnist_pca')', [], 2);
    
    NN_PCA_mnist_error(j) = sum(NN_class ~= y_mnist)/n_mnist;
    j = j + 1;

end

% FASHION

NN_PCA_fashion_error = zeros(length(PCA_sizes),1);
j = 1;

for PC_dim = PCA_sizes

    % PCA of the MNIST data:
    X_fashion_pca = X_fashion * pca(X_fashion);
    X_fashion_pca = X_fashion_pca(:, 1:PC_dim);

    % Neural network:
    net = feedforwardnet([NN_hidden_size, 20, 15], 'trainrp');
    net = train(net, X_fashion_pca',  Y_fashion', [], [], [], 'useParallel');
    [~, NN_class] = max(net(X_fashion_pca')', [], 2);
    
    NN_PCA_fashion_error(j) = sum(NN_class ~= y_mnist)/n_mnist;
    j = j + 1;

end

%% Convolutional Neural Network

X_mnist_datastore   = zeros(20, 20, 1, n_mnist);
X_fashion_datastore = zeros(28, 28, 1, n_mnist);

options = trainingOptions('sgdm');
rng('default')

% MNIST

for j = 1:n_mnist
    X_mnist_datastore(:,:,:,j) = reshape(X_mnist(j,:), 20, 20);
end

CNN_layers = [ ...
                imageInputLayer([20, 20, 1]);
                convolution2dLayer(10, 5);
                reluLayer;
                fullyConnectedLayer(10);
                softmaxLayer();
                classificationLayer();
              ];
          

CNN_net_mnist = trainNetwork(X_mnist_datastore, categorical(y_mnist), ...
                        CNN_layers, options);
                
CNN_predict_mnist = double(classify(CNN_net_mnist, X_mnist_datastore));
CNN_mnist_error = sum(CNN_predict_mnist ~= y_mnist)/n_mnist;

misclass = zeros(10, 10);
for k = 1:10
    for j = 1:10
        misclass(k, j) = sum(CNN_predict_mnist(y_mnist == k) == j)/...
                            (n_mnist/10);
    end
end

figure
imagesc(misclass)
pbaspect([2 2 1])
title('CNN classifier - Fashion - Misclassification matrix')

% FASHION

for j = 1:n_mnist
    X_fashion_datastore(:,:,:,j) = reshape(X_fashion(j,:), 28, 28);
end

CNN_layers = [ ...
                imageInputLayer([28, 28, 1]);
                convolution2dLayer(10, 5);
                reluLayer;
                fullyConnectedLayer(10);
                softmaxLayer();
                classificationLayer();
              ];
          
          
CNN_net_fashion = trainNetwork(X_fashion_datastore, categorical(y_fashion), ...
                        CNN_layers, options);
                
CNN_predict_fashion = double(classify(CNN_net_fashion, X_fashion_datastore));
CNN_fashion_error = sum(CNN_predict_fashion ~= (y_fashion+1))/n_mnist;

misclass = zeros(10, 10);
for k = 1:10
    for j = 1:10
        misclass(k, j) = sum(CNN_predict_fashion((y_fashion+1) == k) == j)/...
                            (n_mnist/10);
    end
end

figure
imagesc(misclass)
pbaspect([2 2 1])
title('CNN classifier - Fashion - Misclassification matrix')



