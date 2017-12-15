% The objective of this exercise is constructing several classifiers for
% anemia using the blood hemoglobin concentrations and the age of the subjects under
% consideration. 
% X1 denotes the concentration of hemoglobin in grams/liter
% X2 denotes the age of the subject
% Y=1 means "anemic", Y=0 means "healthy"
% There are hence only two classes
% Goal: classify data into two classes based on these two features
%
clear
close all

%% The dataset is simulated out of a Gaussian mixture as follows
% Let {X|Y=1}~N([mu_x1_1; mu_x2_1], [sigma2_x1_1 sigma_x1x2_1;sigma_x1x2_1 sigma2_x2_1])
% Let {X|Y=0}~N([mu_x1_0; mu_x2_0], [sigma2_x1_0 sigma_x1x2_0;sigma_x1x2_0 sigma2_x2_0])

% Parameters for "anemic" distribution
mu_x1_1 = 100;
mu_x2_1 = 35;
sigma2_x1_1 = 700;
sigma2_x2_1 = 225;
sigma_x1x2_1 = 100;

% Parameters for "healthy" distribution
mu_x1_0 = 155;
mu_x2_0 = 56;
sigma2_x1_0 = 500;
sigma2_x2_0 = 300;
sigma_x1x2_0 = 150;

% Mixture parameters - feel free to change
p_mixture = 1/4;
w1 = p_mixture;
w0 = 1 - p_mixture;

%% 1.-Simulate a training data set (y_i, x_i) with n elements

n = 100;

mu1 = [mu_x1_1; mu_x2_1];
Sigma1 = [sigma2_x1_1, sigma_x1x2_1; sigma_x1x2_1, sigma2_x2_1];
mu0 = [mu_x1_0; mu_x2_0];
Sigma0 = [sigma2_x1_0, sigma_x1x2_0; sigma_x1x2_0, sigma2_x2_0];

ill_sample = mvnrnd(mu1, Sigma1, n);
healthy_sample = mvnrnd(mu0, Sigma0, n);

mixed_sample = [ill_sample; healthy_sample];
sample_class = [ones(size(ill_sample, 1), 1); ...
                        zeros(size(healthy_sample, 1), 1)];
                    
% Figure limits:
xl = [min(mixed_sample(:,1))-10, max(mixed_sample(:,1))+10];
yl = [min(mixed_sample(:,2))-10, max(mixed_sample(:,2))+10];

%% 2.- Create a 2D scatter plot that represents the hemoglobine level and health of the
% anemic and healthy subjects. Denote the anemic subjects with the symbol '+'
% and with 'o' the healthy ones

figure(1)

subplot(1,2,1)
hold on
title('Separate samples')
plot(ill_sample(:,1), ill_sample(:,2), 'b+')
plot(healthy_sample(:,1), healthy_sample(:,2), 'ro')
grid on
% pbaspect([2 2 1])
legend('ILL sample', 'HEALTHY sample')
hold off

subplot(1,2,2)
hold on
title('Mixed sample')
plot(mixed_sample(:,1), mixed_sample(:,2), 'kx')
grid on
% pbaspect([2 2 1])
legend('MIXED sample')
hold off

%% 3.-Assuming that the DGP is known, construct the Bayes classifier and use it to
% classify the generated sample. Start by constructing a function
% getEtaAnemia2features(x_current, w1, mu_x_1, Sigma_x_1, w0, mu_x_0, Sigma_x_0)
% that computes P(Y=1|X=x) when P(X) is given by a mixture of two
% Gaussians.
% Compute the number of Type I and Type II errors and the
% empirical risk of the Bayes' classifier

eta_f = @(x) getEtaAnemia2features(x, w1, mu1, Sigma1, ...
                                      w0, mu0, Sigma0) - 1/2;

% x0 = min(mu1, mu0) + (mu1-mu0)/2;
% 
% bayes_crit = fzero(eta_f, x0);

bayes_ill = mixed_sample(eta_f(mixed_sample) > 0, :);
bayes_healthy = mixed_sample(eta_f(mixed_sample) <= 0, :);

% Type 1 errors:
Type1_er_bayes = setdiff(bayes_ill, ill_sample, 'rows');
% Type 2 errors:
Type2_er_bayes = setdiff(bayes_healthy, healthy_sample, 'rows');

% Empirical risk:
bayes_emp_risk = (length(Type1_er_bayes) + length(Type2_er_bayes))/n;

%% 4.- Redo the plot in point 2 and mark in red the markers corresponding to the 
% subjects that are misclassified by the Bayes classifier

figure(2)
hold on
title('Bayes classifier error')
plot(ill_sample(:,1), ill_sample(:,2), 'k+')
plot(healthy_sample(:,1), healthy_sample(:,2), 'ko')
plot(Type1_er_bayes(:,1), Type1_er_bayes(:,2), 'rx', 'MarkerSize', 12)
plot(Type2_er_bayes(:,1), Type2_er_bayes(:,2), 'gx', 'MarkerSize', 12)
xlim(xl)
ylim(yl)

bound_image = getDecisionBoundaryPlot(xl, yl, eta_f, 1);
colormap spring
imagesc(xl, yl, bound_image, 'AlphaData', 0.2)

grid on
legend('ILL sample', 'HEALTHY sample', 'Type 1 errors', 'Type 2 errors', ...
            'Location', 'best')
% pbaspect([2 2 1])
% legend('MIXED sample')
hold off

%% 5.- Construct the logistic classifier based on the training sample and
% compute the associated errors and classifier's empirical risk.

logit_crit = @(x, beta) exp([ones(size(x,1),1), x]*beta(:));

log_beta = glmfit(mixed_sample, sample_class, 'binomial', 'link', 'logit');

logit_ill = mixed_sample(logit_crit(mixed_sample, log_beta) > 1, :);
logit_healthy = mixed_sample(logit_crit(mixed_sample, log_beta) <= 1, :);

% Type 1 errors:
Type1_er_logit = setdiff(logit_ill, ill_sample, 'rows');
% Type 2 errors:
Type2_er_logit = setdiff(logit_healthy, healthy_sample, 'rows');

% Empirical risk:
logit_emp_risk = (length(Type1_er_logit) + length(Type2_er_logit))/n;

%% 6.- Redo the plot in point 2 and mark in red the markers corresponding to the 
% subjects that are misclassified by the logistic classifier.

figure(3)
hold on
title('Logistic classifier error')

plot(ill_sample(:,1), ill_sample(:,2), 'k+')
plot(healthy_sample(:,1), healthy_sample(:,2), 'ko')
plot(Type1_er_logit(:,1), Type1_er_logit(:,2), 'rx', 'MarkerSize', 12)
plot(Type2_er_logit(:,1), Type2_er_logit(:,2), 'gx', 'MarkerSize', 12)
xlim(xl)
ylim(yl)

bound_image = getDecisionBoundaryPlot(xl, yl, ...
                                    @(x) logit_crit(x, log_beta)-1, 1);
colormap spring
imagesc(xl, yl, bound_image, 'AlphaData', 0.2)

grid on
legend('ILL sample', 'HEALTHY sample', 'Type 1 errors', 'Type 2 errors', ...
            'Location', 'best')
% pbaspect([2 2 1])
% legend('MIXED sample')
hold off

%% 7.- Construct the LDA classifier based on the training sample and 
% compute the associated errors and classifier's empirical risk.

%% 8.- Construct the QDA classifier based on the training sample and
% compute the associated errors and classifier's empirical risk.

%% 7.- Construct the Naive Bayes classifier based on the training sample and 
% compute the associated errors and classifier's empirical risk.

%% 9.- Construct perceptron hyperplane classifier based on the training sample and 
% compute the classifier's empirical risk.
% The perceptron is specified by a vector beta = [beta0, beta1, beta2] that 
% determines a separating hyperplane beta0 + beta1 * x1 + beta2 * x2
% Start by constructing a function that computes the empirical error of this
% classifier as a function of beta.

% 10.- Classify the data with a KNN classifier and compute the associated errors 
% and classifier's empirical risk with 1, 5, and 10 neighbors

% 11.- Classify the data using the unsupervised approach providing the number of classes 2
% (for that make use of the EM and MLE techniques implemented earlier; for the MLE 
% technique make an improvement and guarantee PSD constraint by construction using the Cholesky
% decomposition of covariance matrix), compute the associated errors and classifier's empirical risk.

% 12.- Generate a new testing sample and compute the testing errors of all the above
% classifiers

% 13.- Plot decision boundaries for the above used classifiers
% For that adapt the code in
% http://www.peteryu.ca/tutorials/matlab/visualize_decision_boundaries 
% for your needs

