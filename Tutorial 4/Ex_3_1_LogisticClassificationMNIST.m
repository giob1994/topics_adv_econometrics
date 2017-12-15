% The goal of this exercise is creating a handwriting recognition
% for a data set that contains handwritten digits between zero and nine 
% using a one-vs-all logistic classifier.
% The handwritten digits are characterized by a 20 x 20 pixel gray level
% matrix that is stored in the lines of the file digits_data.csv. The label
% corresponding to each digit is stored in digits_labels.csv; the digit "0"
% is mapped to label "10" in this list.

clear
close all


%% 1.- Read the data out of the files 'digits_data.csv' and 'digits_labels.csv' 
% and create a design matrix X
% in which the rows contain the pixel gray levels of each image. Each row
% should contain 400 values. Create also a vector y containing the labels
% associated to each picture

X = csvread('digits_data.csv');
Y = csvread('digits_labels.csv');

n = size(X, 1);
p = size(X, 2);


%% 2.- Create a function [h, display_array] = displayData(X, example_width)
% % that displays 2D data in a grid out of the design matrix X that you 
% constructed. It returns the figure handle h and the displayed array if requested.
% Use this function to display 100 randomly chosen figures in the data set.

r = randperm(n);
selected = X(r(1:100), :);

displayData(selected)

s = selected(1,:);

figure
colormap gray
imagesc(reshape(s, round(sqrt(p)), round(sqrt(p))))
pbaspect([2 2 1])
xticks([])
yticks([])


%% 3.- Construct a logistic classifier to find the "1"s in the dataset using 
% the built-in matlab (glmfit) function. Compute the Type I,
% Type II, and empirical errors.

% We try to classify only the '1's:

digit_to_classify = 5;

class_1 = (Y == digit_to_classify);
class_not1 = (Y ~= digit_to_classify);

logit_crit = @(x, beta) exp([ones(size(x,1),1), x]*beta(:)) - 1;

log_beta = glmfit(X, class_1, 'binomial', 'link', 'logit');

logit_1 = (logit_crit(X, log_beta) > 0);
logit_not1 = (logit_crit(X, log_beta) <= 0);

% Display an example digits:
sample_X_logit_1 = X(logit_1, :);
[h, ~] = displayData(sample_X_logit_1(1:100, :));
title('Logit classifier - Sample of 1s')

% Type 1 errors:
Type1_er_logit = sum(logit_1 ~= class_1);
% Type 2 errors:
Type2_er_logit = sum(logit_not1 ~= class_not1);

% Empirical risk:
logit_emp_risk = (Type1_er_logit + Type2_er_logit)/n;


%% 4.- One-vs-all logistic regression. Construct a one-vs-all logistic
% classifier for the 10 figures. Compute the Type I, Type II, and empirical errors.

logit_crit_m = @(x, beta) exp([ones(size(x,1),1), x]*beta(:));

crit_matrix = zeros(size(X, 1), 10);

for k = 0:9
    
    beta_k = glmfit(X, (Y == k), 'binomial', 'link', 'logit');
    
    crit_matrix(:, k+1) = logit_crit_m(X, beta_k);
    
end

[~, logit_1vsall] = max(crit_matrix, [], 2);
logit_1vsall = logit_1vsall-1;
logit_1vsall(logit_1vsall == 0) = 10;

er_logit_1vsall = sum(logit_1vsall ~= Y);

% Plot missclassified digits:
X_errors = X(logit_1vsall ~= Y, :);
Y_errors = logit_1vsall(logit_1vsall ~= Y);
Y_true = Y(logit_1vsall ~= Y);
r = randperm(er_logit_1vsall);
selected_X = X_errors(r(1:16), :);
selected_er = Y_errors(r(1:16));
selected_Y_true = Y_true(r(1:16));

displayData(selected_X);
% [place_x, place_y] = meshgrid((0:1/4:3/4), (0:1/4:3/4));
% place_x = place_x(:);
% place_y = place_y(:);
% x_l = xlim;
% y_l = ylim;
% for i = 1:16
%     text(x_l(1) + 2 + (x_l(2)-x_l(1))*place_x(i), ...
%          y_l(1) + 1 + (y_l(2)-y_l(1))*place_y(i), ...
%          ['Logit: ', num2str(selected_er(i))], 'Color', [1, 0.3, 0]);
%     text(x_l(1) + 12 + (x_l(2)-x_l(1))*place_x(i), ...
%          y_l(1) + 18 + (y_l(2)-y_l(1))*place_y(i), ...
%          ['Class: ', num2str(selected_Y_true(i))], 'Color', 'white');
% end

%% 5.- Create a misclassification matrix whose (i,j)th element
% denotes the percentage of times in which the classifier assigns the 
% figure with label i the label j.

misclass = zeros(10, 10);

for k = 1:10
    for j = 1:10
        
        misclass(k, j) = sum(logit_1vsall(Y == k) == j)/n/10;
        
    end
end

figure
imagesc(misclass)
pbaspect([2 2 1])
title('Logit classifier - Misclassification matrix')

%% 6.- Use the function displayData in point 2 in order to visualize the 
% "7"s that get classified as a "9".

