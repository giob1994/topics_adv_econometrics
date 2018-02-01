% The goal of this exercise is finding the weights in a neural network with the
% architecture used in the exercise Ex_3_2 that minimize the empirical classification error
% in the handwritting recognition problem.
% This will be done by implementing the backpropagation algorithm.

clear;
close all;

% 1.- Read the data out of the mentioned files 'digits_data.csv' and 'digits_labels.csv' 
% and create a design matrix X
% in which the rows contain the pixel gray levels of each image. Each row
% should contain 400 values. Create also a vector y containing the labels
% associated to each picture

X = csvread('digits_data.csv');
Y = csvread('digits_labels.csv');

n = size(X, 1);
p = size(X, 2);

% 2.- Create a function [J grad] = nnminuslogLikelihood(nn_params, input_layer_size, ...
% hidden_layer_size, num_labels, X, y, lambda) that computes the log-likelihood
% of a neural network with one hidden layer as well as its gradient using
% the backpropagation algorithm

% 3.- Check the correctness of you implementation by comparing the gradient output
% of your function nnlogLikelihood to its numerical evaluation using the 
% definition of the derivative. Perform this check using the weights 
% Theta1 and Theta2 stored in the file 'paramsNN.mat'. It is enough to
% compute the first 100 elements

% 4.- Use the function nnminuslogLikelihood to train the neural network to the classification
% of the digits in 'digits_data.csv'. Use the weights 
% Theta1 and Theta2 stored in the file 'paramsNN.mat' as initial values in
% the optimization process.

% 5.- Use the NN obtained to classify the digits in 'digits_data.csv'
% Compute subsequently the empirical error of the classifier and compare the 
% performance with the one exhibited by the initial NN. 

% 6.- Create a misclassification matrix whose (i,j)th element
% denotes the percentage of cases in which the classifier assigns the 
% figure with label i the label j.

% 7.- Use the function displayData to visualize the 
% "7" that get classified as a "9".

