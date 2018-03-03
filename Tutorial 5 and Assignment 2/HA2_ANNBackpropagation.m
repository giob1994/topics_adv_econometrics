% The goal of this exercise is finding the weights in a neural network with the
% architecture used in the exercise Ex_3_2 that minimize the empirical classification error
% in the handwritting recognition problem.
% This will be done by implementing the backpropagation algorithm.

clear;
close all;

sig     = @(z) 1./(1 + exp(-z));
sig_d1  = @(z) sig(z) .* (1-sig(z));

%% 1.- Read the data out of the mentioned files 'digits_data.csv' and 'digits_labels.csv' 
% and create a design matrix X
% in which the rows contain the pixel gray levels of each image. Each row
% should contain 400 values. Create also a vector y containing the labels
% associated to each picture

X = csvread('digits_data.csv');
y = csvread('digits_labels.csv');

n = size(X, 1);
p = size(X, 2);
num_labels = 10;

Y = zeros(n, num_labels);
for j = 1:n
    Y(j, y(j)) = 1;
end

%% 2.- Create a function [J grad] = nnminuslogLikelihood(nn_params, input_layer_size, ...
% hidden_layer_size, num_labels, X, y, lambda) that computes the log-likelihood
% of a neural network with one hidden layer as well as its gradient using
% the backpropagation algorithm

input_layer_size = p+1;
hidden_layer_size = 25+1;

nn_params = rand( input_layer_size*(hidden_layer_size-1) + ...
                  hidden_layer_size*num_labels, 1 );
              
lambda = 0.02;

[J, grad1, grad2] = nnminuslogLikelihood(nn_params, input_layer_size, ...
                            hidden_layer_size, num_labels, X, y, lambda);

%% 3.- Check the correctness of your implementation by comparing the gradient output
% of your function nnlogLikelihood to its numerical evaluation using the 
% definition of the derivative. Perform this check using the weights 
% Theta1 and Theta2 stored in the file 'paramsNN.mat'. It is enough to
% compute the first 100 elements

test_grad = zeros(100,1);

for I = 1:10
    
    nn_params_delta = nn_params;
    nn_params_delta(I) = nn_params(I) + 0.0001;
    [J__, ~, ~] = nnminuslogLikelihood(nn_params_delta, input_layer_size, ...
                            hidden_layer_size, num_labels, X, y, lambda);
    test_grad(I) = (J__ - J) / (0.0001);                   
    
end 

% hidden_params = reshape(nn_params(1:input_layer_size*(hidden_layer_size-1)), ...
%                         hidden_layer_size-1, input_layer_size);
% output_params = reshape(nn_params((input_layer_size*(hidden_layer_size-1) + 1):end), ...
%                         num_labels, hidden_layer_size);
%                     
% a1 = [ones(size(X,1), 1), X];
% a2 = sig([ones(size(X,1), 1), a1 * hidden_params']);
% y_out = sig(a2 * output_params');
% 
% for I = 1:10
%     for J = 1:10
%         
%         Temp = 0;
%         
%         % For every obs. in the sample:                
%         for i = 1:n
%             % For every output label:
%             for k = 1:num_labels
% 
%                 dAk = sig_d1( [1, sig([1, X(i,:)] * hidden_params')] * output_params(k,:)' ) * ...
%                         output_params(k,J) * sig_d1( [1, X(i,:)] * hidden_params(J,:)' ) * X(i,I);
%                     
%                 Temp = Temp + ...
%                        Y(n, k) * 1/(y_out(n,k)) * dAk - ...
%                        (1 - Y(n,k)) * (1/(1 - y_out(n,k)) * dAk);
% 
%             end 
%         end
%         
%         dJ_dtheta(I,J) =  -1/n * Temp + lambda/n *  hidden_params(I,J);
%     
%     end
% end

%% 4.- Use the function nnminuslogLikelihood to train the neural network to the classification
% of the digits in 'digits_data.csv'. Use the weights 
% Theta1 and Theta2 stored in the file 'paramsNN.mat' as initial values in
% the optimization process.

load('paramsNN.mat')

alpha = 0.2;

params0_1 = Theta1;
params0_2 = Theta2;
params0 = [params0_1(:); params0_2(:)];

diff = 1;

while diff > 10^-7
    
    [J, grad1, grad2] = nnminuslogLikelihood(params0, input_layer_size, ...
                            hidden_layer_size, num_labels, X, y, lambda);
                        
    params1_1 = params0_1 - alpha * grad1;
    params1_2 = params0_2 - alpha * grad2;
    params1 = [params1_1(:); params1_2(:)];
    
    diff = norm(params0 - params1);
    
    params0 = params1;
    
end

% options = optimoptions(--- 'GradObj')

%% 5.- Use the NN obtained to classify the digits in 'digits_data.csv'
% Compute subsequently the empirical error of the classifier and compare the 
% performance with the one exhibited by the initial NN. 

A1 = [ones(size(X,1), 1), X];
A2 = sig([ones(size(X,1), 1), A1 * params1_1']);
A3 = sig(A2 * params1_2');

[~, NN_class] = max(A3, [], 2);

NN_emp_error = sum(NN_class ~= y)/n;

%% 6.- Create a misclassification matrix whose (i,j)th element
% denotes the percentage of cases in which the classifier assigns the 
% figure with label i the label j.

misclass = zeros(10, 10);

for k = 1:10
    for j = 1:10
        
        misclass(k, j) = sum(NN_class(y == k) == j)/(n/10);
        
    end
end

figure
imagesc(misclass)
pbaspect([2 2 1])
title('ANN classifier - Misclassification matrix')

%% 7.- Use the function displayData to visualize the 
% "7" that get classified as a "9".

misclass_7_to_9 = logical( (y == 7) .* (NN_class(:) == 9) );
misclass_7_to_9_digit = X(misclass_7_to_9, :);

displayData(misclass_7_to_9_digit)
title('Miscalssification - digit "7" as "9"')