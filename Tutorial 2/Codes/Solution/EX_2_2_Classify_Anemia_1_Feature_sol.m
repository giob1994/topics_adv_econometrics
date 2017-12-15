% The objective of this exercise is constructing several classifiers for
% anemia using the blood hemoglobin concentrations of the subjects under
% consideration. 
% X denotes the concentration of hemoglobin in grams/liter
% Y=1 means "anemic", Y=0 means "healthy"
% There are hence only two classes
% Goal: classify data into two classes based on one feature
%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% The dataset is simulated out of a Gaussian mixture as follows
% Let {X|Y=1}~N(mu1, sigma12)
% Let {X|Y=0}~N(mu0, sigma02)

% Variables:
% n sample size
% mu1, mu0 means of the Gaussians used in the mixture
% sigma12, sigma02 variances of the Gaussians used in the mixture
% w1, w0 weights for each component of the mixture

clear
close all
%rng default;  % For reproducibility
p_mixture = 1/3;
w1 = p_mixture;
w0 = 1 - p_mixture;

% parameters for the class Y = 1
% mean
mu1 = 100;
% variance
sigma12 = 900;

% parameters for the class Y = 0
% mean
mu0 = 150;
% variance
sigma02 = 500;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% 1.-Simulate a training data set (x_i, y_i) with n elements 
% using Matlab binornd function

n = 2000;
Y = binornd(1, p_mixture, n, 1);
n_sick = sum(Y);
n_healthy = n - n_sick;

Sick_sample_x = mu1 + sqrt(sigma12) * randn(n, 1);
Healthy_sample_x = mu0 + sqrt(sigma02) * randn(n, 1);
Sick_sample_x = Sick_sample_x(Y == 1);
Healthy_sample_x = Healthy_sample_x(Y == 0);
Total_sample_x = [Sick_sample_x; Healthy_sample_x];

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% 2.-In the same figure plot:
% (1)-A count density-normalized histogram of the total subjects sample
% (The height of each bar is number of observations in bin / width of bin. 
% The area (height * width) of each bar is the number of observations in the bin. 
% The sum of the bar areas is the sample size)  
% (2)-A count density-normalized histogram of the healthy subjects sample 
% (3)-A count density-normalized histogram of the sick subjects sample
% (4)-Corresponding weighted versions of the two Gaussians used in the generation
% (5)-Corresponding weighted version of the Gaussian mixture distribution

mean_mixture = w1 * mu1 + w0 * mu0;
sigma2_mixture = w1 * ((mu1 - mean_mixture)^2 + sigma12) + w0 * ((mu0 - mean_mixture)^2 + sigma02);

figure(1)
% Plot histogram of all subjects
h_total_sample_x = histogram(Total_sample_x, 'Normalization','countdensity', 'EdgeColor', 'blue', 'FaceColor', 'blue', 'BinWidth', 10, 'FaceAlpha', .4);
title('Histogram of the total sample of sick and healthy subjects')

hold on
% Plot histogram of sick subjects
h_sick = histogram(Sick_sample_x, 'Normalization','countdensity', 'EdgeColor', 'black', 'FaceColor', 'red', 'BinWidth', 10, 'FaceAlpha', .4);
hold on
% Plot histogram of healthy subjects
h_healthy = histogram(Healthy_sample_x, 'Normalization','countdensity', 'EdgeColor', 'blue', 'FaceColor', 'blue', 'BinWidth', 10, 'FaceAlpha', .4);
hold on
x = linspace(mean_mixture - 3 * sqrt(sigma2_mixture), mean_mixture + 3 * sqrt(sigma2_mixture), 100);
% Plot the "sick" Gaussian weighted with the sample size
y1 = length(Sick_sample_x) * (1 / sqrt(2 * pi * sigma12)) * exp(-(x - mu1).^2/(2 * sigma12));
plot(x,y1, 'LineWidth',2)
% Plot the second Gaussian weighted weighted with the sample size
y2 = length(Healthy_sample_x) *(1 / sqrt(2 * pi * sigma02)) * exp(-(x - mu0).^2/(2 * sigma02));
plot(x,y2, 'LineWidth',2)
% Plot the Gaussian Mixture
y = (length(Healthy_sample_x) + length(Sick_sample_x)) * (w1 *(1 / sqrt(2 * pi * sigma12)) * exp(-(x - mu1).^2/(2 * sigma12)) +...
    w0 *(1 / sqrt(2 * pi * sigma02)) * exp(-(x - mu0).^2/(2 * sigma02)));
plot(x,y, 'LineWidth',2)

title('Histograms with the samples of sick and healthy subjects')
xlim([mean_mixture - 3 * sqrt(sigma2_mixture),mean_mixture + 3 * sqrt(sigma2_mixture)])

hold off

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% 3.(1)-Assuming that the data generating process is known, construct and represent 
% graphically the function eta that determines the Bayes classifier
% 3.(2)-Determine the critical value of the Bayes classifier by determining the
% hemoglobine concentration x_critical that yields eta(x_critical).

eta = @(x) w1 * (1 / sqrt(2 * pi * sigma12)) * exp(-(x - mu1).^2/(2 * sigma12)) ./...
    (w1 *(1 / sqrt(2 * pi * sigma12)) * exp(-(x - mu1).^2/(2 * sigma12)) +...
    w0 *(1 / sqrt(2 * pi * sigma02)) * exp(-(x - mu0).^2/(2 * sigma02)));
eta_x = eta(x);

figure(2);
plot(x,eta_x, 'LineWidth',2)
title('eta function for the Bayes classifier')
xlim([mean_mixture - 3 * sqrt(sigma2_mixture),mean_mixture + 3 * sqrt(sigma2_mixture)])
hold off;

eta_critical = @(x) eta(x) - 1/2;
x_critical_Bayes = fzero(eta_critical, 110);

% 3.(3)-Compute the Bayes' risk according to formulas from Lecture 2
% Was left up to you. Please fill in.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% 4.- Depict the critical value (decision boundary)
% in the first figure
figure(1)
line([x_critical_Bayes x_critical_Bayes],[0 1.5], 'LineWidth', 2, 'Color', 'red');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% 5.- Compute the number of Type I and Type II errors and the
% empirical risk of the Bayes classifier

TypeI_errors_Bayes = length(find(Healthy_sample_x < x_critical_Bayes));
TypeII_errors_Bayes = length(find(Sick_sample_x > x_critical_Bayes));
Empirical_error_Bayes = (TypeI_errors_Bayes + TypeII_errors_Bayes) / n;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% 6.-Demonstrate that empirical risk of Bayes' classifier is an unbiased estimate
% of true Bayes' risk

% let now compute the empirical risk over n_EM_eval of training samples
% generate them m times and compute the mean
m = 1000;
Empirical_error_Bayes_EM_eval = zeros(m, 1);
n_EM_eval = 100;

for j = 1:m
    Y_EM_eval = binornd(1, p_mixture, n_EM_eval, 1);
    Sick_sample_x_EM_eval = mu1 + sqrt(sigma12) * randn(n_EM_eval, 1);
    Healthy_sample_x_EM_eval = mu0 + sqrt(sigma02) * randn(n_EM_eval, 1);
    Sick_sample_x_EM_eval = Sick_sample_x_EM_eval(Y_EM_eval == 1);
    Healthy_sample_x_EM_eval = Healthy_sample_x_EM_eval(Y_EM_eval == 0);
    Total_sample_x_EM_eval = [Sick_sample_x_EM_eval; Healthy_sample_x_EM_eval];

    TypeI_errors_Bayes_EM_eval = length(find(Healthy_sample_x_EM_eval < x_critical_Bayes));
    TypeII_errors_Bayes_EM_eval = length(find(Sick_sample_x_EM_eval > x_critical_Bayes));
    % Empirical Bayes risk evaluation over number n_EM_eval of training
    % examples
    Empirical_error_Bayes_EM_eval(j) = (TypeI_errors_Bayes_EM_eval + TypeII_errors_Bayes_EM_eval) / n_EM_eval;
end
% one needs to average on many instances
Empirical_error_Bayes_EM_eval_res = mean(Empirical_error_Bayes_EM_eval);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Practical classifiers

% 7.- Construct the LDA classifier associated to the previous sample and 
% compute the associated errors and classifier risk

n_1 = length(Sick_sample_x);
n_2 = length(Healthy_sample_x);
pi_1 = n_1 / n;
pi_2 = n_2 / n;
mu_1 = mean(Sick_sample_x);
mu_2 = mean(Healthy_sample_x);
sigma_hat2 = (1 / (n-2)) * (sum((Sick_sample_x - mu_1).^2) + sum((Healthy_sample_x - mu_2).^2));
x_critical_LDA = (sigma_hat2/(mu_1 - mu_2)) * (((mu_1^2 - mu_2^2)/(2*sigma_hat2)) + log(pi_2/pi_1));
TypeI_errors_LDA = length(find(Healthy_sample_x < x_critical_LDA));
TypeII_errors_LDA = length(find(Sick_sample_x > x_critical_LDA));
Empirical_error_LDA = (TypeI_errors_LDA + TypeII_errors_LDA) / n;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% 8.- Construct the logistic classifier associated to the previous sample and 
% compute the associated errors and classifier risk.

[logitCoef,dev] = glmfit([Sick_sample_x; Healthy_sample_x],[ones(n_1, 1); zeros(n_2, 1)],'binomial','logit');
x_critical_logistic = - logitCoef(1)/logitCoef(2);
TypeI_errors_logistic = length(find(Healthy_sample_x < x_critical_logistic));
TypeII_errors_logistic = length(find(Sick_sample_x > x_critical_logistic));
Empirical_error_logistic = (TypeI_errors_logistic + TypeII_errors_logistic) / n;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% 9.- Unsupervised learning. Now we do not assume 
% that labels are observed anymore. The only sample which is available is 
% the whole features sample x but we assume that the underlying
% distribution is a Gaussian mixture with two components. Construct a
% classifier based on fitting such a distribution to the observed features
% using the technique introduced in Ex_1_1_GaussianMix_sol_withEM. 
% You need both to use the MLE and EM techniques. Compute the associated errors 
% and classifier empirical risk

