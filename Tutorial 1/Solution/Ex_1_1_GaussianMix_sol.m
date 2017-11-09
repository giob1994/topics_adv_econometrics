% The objective of this exercise is simulating out of a Gaussian mixture
% model and fitting a Gaussian mixture distribution to a
% data set by maximizing the log-likelihood, using Expectation-Maximization
% (EM)  algorithm and built-in matlab function with implemented EM

% We consider first the case of mixture of one-dimensional Gaussian distributions.
% By the next tutorial you need to extend the codes for the case of
% multivariate Gaussian mixtures with arbitrary number of components.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Variables:
% n sample size
% mu1, mu2 means of the Gaussians used in the mixture
% sigma12, sigma22 variances of the Gaussians used in the mixture
% w1, w2 weights for each component of the mixture

clear
close all
rng default;  % For reproducibility
n = 50;
p_mixture = 1/3;

w1 = p_mixture;
w2 = 1 - p_mixture;

% parameters for the first component
% mean
mu1 = 3;
% variance
sigma12 = 1;

% parameters for the second component
% mean
mu2 = -3;
% variance
sigma22 = 2;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% 1.-Generate a Gaussian mixture distributed sample of size n of means mu1, mu2,
% variances sigma12 and sigma22, and mixing weights w1, w2
% (a) using no built-in special functions
% (b) using Matlab functions: gmdistribution, binornd

% there are different methods to simulate:
% (1) specify the exact length of each sample according to the weight of the
% component, namely w1 = p_mixture and w2 = 1 - p_mixture
n_component1 = floor(n * w1);
n_component2 = n - n_component1;
% generate exactly the samples with lengths n_component1 and n_component2
Gaussian_sample1 = mu1 + sqrt(sigma12) * randn(n_component1, 1);
Gaussian_sample2 = mu2 + sqrt(sigma22) * randn(n_component2, 1);
GaussMix_sample = [Gaussian_sample1; Gaussian_sample2];

% (2) specify the exact length of each sample according to the weight of the
% component, namely w1 = p_mixture and w2 = 1 - p_mixture
% generate the samples with length n and take from them the needed
% subsamples
n_component1 = floor(n * w1);
n_component2 = n - n_component1;
Gaussian_sample1 = mu1 + sqrt(sigma12) * randn(n, 1);
Gaussian_sample2 = mu2 + sqrt(sigma22) * randn(n, 1);
Gaussian_sample1 = Gaussian_sample1(1:n_component1);
Gaussian_sample2 = Gaussian_sample2(1:n_component2);
GaussMix_sample = [Gaussian_sample1; Gaussian_sample2];

% The following 3 methods allow to simulate the hierarchical model, namely:
% Y~Bernoulli(p)
% X|Y=1~N(mu1, sigma12)
% X|Y=0~N(mu2, sigma22)
% Think - why (1) and (2) are different from (3)-(5). To be discussed in
% what follows

% (3) let the labels of components be also
% randomly distributed; generate uniform indexes of length n,
% generate the samples with length n and take from them the needed
% subsamples based on indexes exceeding or not p_mixture (this allows to
% serve with the standard technique of X~Bernoulli(p) simulation, namely:
% let U~Uniform(0,1); if U<=p then X=1 else X=0)
Indexes = rand(n, 1);
Gaussian_sample1 = mu1 + sqrt(sigma12) * randn(n, 1);
Gaussian_sample2 = mu2 + sqrt(sigma22) * randn(n, 1);
Gaussian_sample1 = Gaussian_sample1(Indexes <= p_mixture);
Gaussian_sample2 = Gaussian_sample2(Indexes > p_mixture);
n_component1 = length(Gaussian_sample1);
n_component2 = length(Gaussian_sample2);
GaussMix_sample = [Gaussian_sample1; Gaussian_sample2];

% (4) let the labels of components be also
% randomly distributed as Bernoulli(p_mixture). The same way as in (3) 
% but with the built-in function 
Y = binornd(1, p_mixture, n, 1);
Gaussian_sample1 = mu1 + sqrt(sigma12) * randn(n, 1);
Gaussian_sample2 = mu2 + sqrt(sigma22) * randn(n, 1);
Gaussian_sample1 = Gaussian_sample1(Y == 1);
Gaussian_sample2 = Gaussian_sample2(Y == 0);
n_component1 = sum(Y);
n_component2 = n - n_component1;
GaussMix_sample = [Gaussian_sample1; Gaussian_sample2];

% (5) use Matlab function to generate the sample
mu = [mu1; mu2];
sigma = cat(3, (sigma12), (sigma22));
p = [w1; w2];
obj = gmdistribution(mu, sigma, p);
GaussMix_sample = random(obj, n);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% 2.-Plot a pdf-normalized histogram of the simulated data together with weighted
% versions of the two Gaussians used in the simulation, and the Gaussian
% mixture distribution; all the pdf's need to be written explicitly

mean_mixture = w1 * mu1 + w2 * mu2;
sigma2_mixture = w1 * ((mu1 - mean_mixture)^2 + sigma12) + w2 * ((mu2 - mean_mixture)^2 + sigma22);

figure(1)

h = histogram(GaussMix_sample, 'Normalization', 'pdf');

hold on

x = linspace(mean_mixture - 3 * sqrt(sigma2_mixture), mean_mixture + 3 * sqrt(sigma2_mixture), 100);
% Plot the first Gaussian weighted
y1 = w1 * (1 / sqrt(2 * pi * sigma12)) * exp(-(x - mu1).^2 / (2 * sigma12));
plot(x, y1, 'r', 'LineWidth', 2)
% Plot the second Gaussian weighted
y2 = w2 * (1 / sqrt(2 * pi * sigma22)) * exp(-(x - mu2).^2 / (2 * sigma22));
plot(x, y2, 'g', 'LineWidth', 2)
% Plot the Gaussian Mixture
y = w1 * (1 / sqrt(2 * pi * sigma12)) * exp(-(x - mu1).^2 / (2 * sigma12)) +...
    w2 * (1 / sqrt(2 * pi * sigma22)) * exp(-(x - mu2).^2 / (2 * sigma22));
plot(x, y, 'b', 'LineWidth',2)

title('Gaussian components and Gaussian mixture for the sample')
xlim([mean_mixture - 3 * sqrt(sigma2_mixture),mean_mixture + 3 * sqrt(sigma2_mixture)])
legend('histogram normalized by pdf', 'component 1', 'component 2', 'Gaussian mixture');

hold off

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% 3.-Make previous code shorter by creating a function for normal pdf or by
% using the built-in function of matlab "normpdf" and "pdf" with Gaussian
% mixture distribution used

mean_mixture = w1 * mu1 + w2 * mu2;
sigma2_mixture = w1 * ((mu1 - mean_mixture)^2 + sigma12) + w2 * ((mu2 - mean_mixture)^2 + sigma22);

figure(2)

h = histogram(GaussMix_sample, 'Normalization', 'pdf');

hold on

x = linspace(mean_mixture - 3 * sqrt(sigma2_mixture), mean_mixture + 3 * sqrt(sigma2_mixture), 100);
% Plot the first Gaussian weighted
y1 = w1 * normpdf(x, mu1, sqrt(sigma12));
plot(x, y1, 'r', 'LineWidth', 2)
% Plot the second Gaussian weighted
y2 = w2 * normpdf(x, mu2, sqrt(sigma22));
plot(x, y2, 'g', 'LineWidth', 2)
% Plot the Gaussian Mixture
y = pdf(obj, x');
plot(x, y, 'b', 'LineWidth',2)

title('Gaussian components and Gaussian mixture for the sample')
xlim([mean_mixture - 3 * sqrt(sigma2_mixture),mean_mixture + 3 * sqrt(sigma2_mixture)])
legend('histogram normalized by pdf', 'component 1', 'component 2', 'Gaussian mixture');

hold off

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% 4.- vary sample lengths and moments of ditributions to see 
% - the cases of distanced centers of Gaussian mixture components and closer ones;
% - disbalanced lengths of samples
% Additionally, check the need for normalization in the histogram
% this part is left for you

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% 5.-Create a function getGaussianMixLogLikBicomp(x, theta) that computes the
% Gaussian Mixture Log-likelihood for 2 components for a sample x with means mu1 = theta(1),
% mu2 = theta(2), variances sigma12 = theta(3), sigma22 = theta(4), and 
% weights w1 = theta(5), w2 = theta(6);

theta = [mu1, mu2, sigma12, sigma22, w1, w2];
loglikValBiComp = getGaussianMixLogLikBicomp(GaussMix_sample, theta);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% 6.-Create a function getGaussianMixLogLikelihood(x, theta) that evaluates the
% Gaussian Mixture Log-likelihood for any number of components at a sample x 
% with the vector theta of concatenated means mu, variances sigma2, and weights w
% verify that it works the same as the previous function provided that the
% number of components is 2

mu = [mu1; mu2];
sigma2 = [sigma12; sigma22];
w = [w1; w2];
theta = [mu; sigma2; w];
loglikValBiCompCheck = getGaussianMixLogLikelihood(GaussMix_sample, theta);

% check if the code is working for the three components
mu = [mu1; mu2; mu2 + 2];
sigma2 = [sigma12; sigma22; 3];
w = [1/3; 1/3; 1/3];
theta = [mu; sigma2; w];
% change sample for that ! but we compute for the one we generated to simplify things 
loglikVal = getGaussianMixLogLikelihood(GaussMix_sample, theta);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% 7.-Fit a Gaussian Mixture to the simulated sample by maximizing the log-likelihood
% Enforce in the maximization process the positivity of the variances and
% that the sum of weights equals one

funToMinimize = @(theta) -getGaussianMixLogLikBicomp(GaussMix_sample, theta);
options = optimoptions(@fmincon,'Display','iter');
Aeq = [0 0 0 0 1 1];
beq = 1;
[theta_mle, loglike1] = fmincon(funToMinimize,[0 0 1 1 0.5 0.5],[],[],Aeq,beq,[-Inf -Inf 0 0 0 0],[Inf Inf Inf Inf 1 1],[],options);
mu1_mle = theta_mle(1);
mu2_mle = theta_mle(2);
sigma12_mle = theta_mle(3);
sigma22_mle = theta_mle(4);
w1_mle = theta_mle(5);
w2_mle = theta_mle(6);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% 8.-Plot in the same figure the original (used for data generation) and the 
% ML-estimated Gaussian mixture.
% Experiment and observe how they become closer as the the sample size grows

figure(2)

hold on

x = linspace(mean_mixture - 3 * sqrt(sigma2_mixture), mean_mixture + 3 * sqrt(sigma2_mixture), 100);
% Plot the original Gaussian mixture
y = w1 * (1 / sqrt(2 * pi * sigma12)) * exp(-(x - mu1).^2/(2 * sigma12)) +...
    w2 * (1 / sqrt(2 * pi * sigma22)) * exp(-(x - mu2).^2/(2 * sigma22));
plot(x, y, 'r', 'LineWidth',2)
% Plot the estimated Gaussian mixture
y_mle = w1_mle * (1 / sqrt(2 * pi * sigma12_mle)) * exp(-(x - mu1_mle).^2/(2 * sigma12_mle)) +...
    w2_mle * (1 / sqrt(2 * pi * sigma22_mle)) * exp(-(x - mu2_mle).^2/(2 * sigma22_mle));
plot(x, y_mle, 'b', 'LineWidth',2)
title('True Gaussian mixture and Gaussian mixture fit for the sample')
xlim([mean_mixture - 3 * sqrt(sigma2_mixture), mean_mixture + 3 * sqrt(sigma2_mixture)])
legend('true', 'estimated');
hold off

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% 9.-Use matlab function to MG to x fixing number of components k = 2;

rng(1); % Reset seed for common start values
% fit Gaussian mixture
Options = statset('Display', 'final', 'MaxIter', 1500, 'TolFun', 1e-5);
n_components = 2;
try
    GMModel = fitgmdist(GaussMix_sample, n_components, 'Options', Options);
catch exception
    disp('There was an error fitting the Gaussian mixture model');
    error = exception.message;
end
mu1_fitgm = GMModel.mu(1);
mu2_fitgm = GMModel.mu(2);
sigma12_fitgm = GMModel.Sigma(:, :, 1);
sigma22_fitgm = GMModel.Sigma(:, :, 2);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% 10.-Use your own EM algorithm to estimate the Gaussian mixture model. EM
% is used in fitgmdist but the goal is to implement it on our own


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% 11.-Extend the codes to the case of bivariate Gaussian till the next
% tutorial