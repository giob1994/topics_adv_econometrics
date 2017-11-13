% The objective of this exercise is simulating out of a Gaussian mixture
% model and fitting a Gaussian mixture distribution to a
% data set by maximizing the log-likelihood, using Expectation-Maximization
% (EM)  algorithm and built-in matlab function with implemented EM

% We consider first the case of mixture of one-dimensional Gaussian distributions.
% By the next tutorial you need to extend the codes for the case of
% multivariate Gaussian mixtures with arbitrary number of components.


% Variables:
% n sample size
% mu1, mu2 means of the Gaussians used in the mixture
% sigma12, sigma22 variances of the Gaussians used in the mixture
% w1, w2 weights for each component of the mixture

clear 
close all
rng default;


% 1.-Generate a Gaussian mixture distributed sample of size n of means mu1, mu2,
% variances sigma12 and sigma22, and mixing weights w1, w2
% (1) using no built-in special functions
% (2) using Matlab function gmdistribution

n = 1000;
p_mixture = 1/3;

w1 = p_mixture;
w2 = 1 - p_mixture;

mu1 = 3;
sigma12 = 1;

mu2 = -3;
sigma22 = 2;

% method 1: decide the exact length of each sample (according to p_mixture)
%           and the generate the gaussians;
% method 2: 

% generate the sample by first constructing the Bernoulli and then
% the Gaussians:
bern = binornd(1, p_mixture, n, 1);

gauss_sample_1 = mu1 + sqrt(sigma12) * randn(n,1);
gauss_sample_2 = mu2 + sqrt(sigma22) * randn(n,1);

sample = zeros(n, 1);
sample(bern == 1) = gauss_sample_1(bern == 1);
sample(bern == 0) = gauss_sample_2(bern == 0);

n_component1 = sum(bern);
n_component2 = n - n_component1;

mu = [mu1; mu2];
sigma = cat(3, (sigma12), (sigma22));
p = [w1; w2];
obj = gmdistribution(mu, sigma, p);
Gauss_mix_sample = random(obj, n);


% 2.-Plot a pdf-normalized histogram of the simulated data together with weighted
% versions of the two Gaussians used in the simulation, and the Gaussian
% mixture distribution; all the pdf's need to be written explicitly

mean_mixture = w1*mu1 + w2*mu2;
sigma2_mixture = w1*((mu1 - mean_mixture)^2+sigma12) + ...
                    w2*((mu2 - mean_mixture)^2+sigma22);

figure(1)
hold on

granularity = 100;

h = histogram(Gauss_mix_sample, floor(granularity/3), 'Normalization', 'pdf');

% Plot both the Gaussian pdf with realtive weight:
pdf_1 = makedist('Normal', mu1, sqrt(sigma12));
pdf_2 = makedist('Normal', mu2, sqrt(sigma22));

x = linspace(mean_mixture - 3*sqrt(sigma2_mixture), mean_mixture + 3*sqrt(sigma2_mixture), granularity);

y1 = w1*pdf(pdf_1, x);
y2 = (1-w1)*pdf(pdf_2, x);

y_mixed = y1 + y2;

plot(x, y1, 'LineWidth', 2)
plot(x, y2, 'LineWidth', 2)
plot(x, y_mixed, 'LineWidth', 2)

hold off


% 3.-Make previous code shorter by creating a function for normal pdf or by
% using the built-in function of matlab "normpdf" and "pdf" with Gaussian
% mixture distribution used

% already done


% 4.- vary sample lengths and moments of ditributions to see 
% - the cases of distanced centers of Gaussian mixture components and closer ones;
% - disbalanced lengths of samples
% Additionally, check the need for normalization in the histogram
% this part is left for you

% Case 1 - means are close to each other:

n = 8000;
p_mixture = 1/3;

w1 = p_mixture;
w2 = 1 - p_mixture;

% mu1 = 3.4;
% sigma12 = 1.2;
% 
% mu2 = 1.2;
% sigma22 = 3;

mu1 = 3;
sigma12 = 1;

mu2 = -0.5;
sigma22 = 2;

mu = [mu1; mu2];
sigma = cat(3, (sigma12), (sigma22));
p = [w1; w2];
obj = gmdistribution(mu, sigma, p);
Gauss_mix_sample = random(obj, n);

mean_mixture = w1*mu1 + w2*mu2;
sigma2_mixture = w1*((mu1 - mean_mixture)^2+sigma12) + ...
                    w2*((mu2 - mean_mixture)^2+sigma22);
                
figure(2)
hold on

granularity = 100;

h = histogram(Gauss_mix_sample, floor(granularity/3), 'Normalization', 'pdf');

% Plot both the Gaussian pdf with realtive weight:
pdf_1 = makedist('Normal', mu1, sqrt(sigma12));
pdf_2 = makedist('Normal', mu2, sqrt(sigma22));

x = linspace(mean_mixture - 3*sqrt(sigma2_mixture), mean_mixture + 3*sqrt(sigma2_mixture), granularity);

y1 = w1*pdf(pdf_1, x);
y2 = w2*pdf(pdf_2, x);

y_mixed = y1 + y2;

plot(x, y1, 'LineWidth', 2)
plot(x, y2, 'LineWidth', 2)
plot(x, y_mixed, 'LineWidth', 2)

hold off

% 5.-Create a function getGaussianMixLogLikBicomp(x, theta) that computes the
% Gaussian Mixture Log-likelihood for 2 components for a sample x with means mu1 = theta(1),
% mu2 = theta(2), variances sigma12 = theta(3), sigma22 = theta(4), and 
% weights w1 = theta(5), w2 = theta(6);

theta = [mu1, mu2, sigma12, sigma22, w1, w2];

loglikeValBiComp = getGaussianMixLogLikBicomp(Gauss_mix_sample, theta);

% 6.-Create a function getGaussianMixLogLikelihood(x, theta) that evaluates the
% Gaussian Mixture Log-likelihood for any number of components at a sample x 
% with the vector theta of concatenated means mu, variances sigma2, and weights w
% verify that it works the same as the previous function provided that the
% number of components is 2

loglikeValBiCompCheck = getGaussianMixLogLikelihood(Gauss_mix_sample, theta);

if (loglikeValBiComp == loglikeValBiCompCheck)
    disp(' [ Log-likelihood calculations are consistent ]')
else
    error(' [ "getGaussianMixLogLikelihood( )" incorrectly defined ]')
end

% 7.-Fit a Gaussian Mixture to the simulated sample by maximizing the log-likelihood
% Enforce in the maximization process the positivity of the variances and
% that the sum of weights equals one

loglikeF = @(theta) -getGaussianMixLogLikBicomp(Gauss_mix_sample, theta);
options = optimoptions(@fmincon, 'Display', 'iter');

x0 = [0, 0, 1, 1, 1/2, 1/2];
Aeq = [0, 0, 0, 0, 1, 1];
beq = 1;
lb = [-Inf, -Inf, 0, 0, 0, 0];
ub = [Inf, Inf, Inf, Inf, 1, 1];

[theta_mle, log_like_mle] = ...
    fmincon(loglikeF, x0, [], [], Aeq, beq, lb, ub, [], options);

mu1_mle = theta_mle(1);
mu2_mle = theta_mle(2);
sigma12_mle = theta_mle(3);
sigma22_mle = theta_mle(4);
w1_mle = theta_mle(5);
w2_mle = theta_mle(6);

% 8.-Plot in the same figure the original (used for data generation) and the ML-estimated Gaussian mixture
% Experiment and observe how they become closer as the the sample size grows

figure(3)

hold on

granularity = 100;

h = histogram(Gauss_mix_sample, floor(granularity/3), 'Normalization', 'pdf');
alpha(h, 0.2)

% Plot both the Gaussian pdf with realtive weight:
pdf_1 = makedist('Normal', mu1, sqrt(sigma12));
pdf_2 = makedist('Normal', mu2, sqrt(sigma22));

x = linspace(mean_mixture - 3*sqrt(sigma2_mixture), mean_mixture + 3*sqrt(sigma2_mixture), granularity);

y1 = w1*pdf(pdf_1, x);
y2 = w2*pdf(pdf_2, x);

y_mixed = y1 + y2;

p1 = plot(x, y1, 'LineWidth', 1.5, 'LineStyle', '--');
p2 = plot(x, y2, 'LineWidth', 1.5, 'LineStyle', '--');
p3 = plot(x, y_mixed, 'LineWidth', 1.5, 'LineStyle', '--');

% Plot the MLE Gaussian pdfs with relative weigth:
pdf_1_mle = makedist('Normal', mu1_mle, sqrt(sigma12_mle));
pdf_2_mle = makedist('Normal', mu2_mle, sqrt(sigma22_mle));

y1_mle = w1_mle*pdf(pdf_1_mle, x);
y2_mle = w2_mle*pdf(pdf_2_mle, x);
y_mixed_mle = y1_mle + y2_mle;

p1_ = plot(x, y1_mle, 'LineWidth', 2);
p2_ = plot(x, y2_mle, 'LineWidth', 2);
p3_ = plot(x, y_mixed_mle, 'LineWidth', 2);

hold off

% 9.-Use matlab function fitgmdist to fit a Gaussian Mixture to the simulated sample

rng(1)

k = 2;
ops = statset('Display','final','MaxIter',1500,'TolFun',1e-5);

GMModel = fitgmdist(Gauss_mix_sample, k, 'Options', ops);

mu1_fitgm = GMModel.mu(1);
mu2_fitgm = GMModel.mu(2);
sigma12_fitgm = GMModel.Sigma(:, :, 1);
sigma22_fitgm = GMModel.Sigma(:, :, 2);

% 10.-Use your own EM algorithm to estimate the Gaussian mixture model. EM
% is used in fitgmdist but the goal is to implement it on our own

precision = 0.1;

theta_em = fitMixedGaussianEMmono(Gauss_mix_sample, theta, precision);

mu1_em = theta_em(1);
mu2_em = theta_em(2);
sigma12_em = theta_em(3);
sigma22_em = theta_em(4);
w1_em = theta_em(5);
w2_em = theta_em(6);

figure(4)

hold on

granularity = 100;

h = histogram(Gauss_mix_sample, floor(granularity/3), 'Normalization', 'pdf');
alpha(h, 0.2)

% Plot both the Gaussian pdf with realtive weight:
pdf_1 = makedist('Normal', mu1, sqrt(sigma12));
pdf_2 = makedist('Normal', mu2, sqrt(sigma22));

x = linspace(mean_mixture - 3*sqrt(sigma2_mixture), mean_mixture + 3*sqrt(sigma2_mixture), granularity);

y1 = w1*pdf(pdf_1, x);
y2 = w2*pdf(pdf_2, x);

y_mixed = y1 + y2;

p1 = plot(x, y1, 'LineWidth', 1.5, 'LineStyle', '--');
p2 = plot(x, y2, 'LineWidth', 1.5, 'LineStyle', '--');
p3 = plot(x, y_mixed, 'LineWidth', 1.5, 'LineStyle', '--');

% Plot the MLE Gaussian pdfs with relative weigth:
pdf_1_em = makedist('Normal', mu1_em, sqrt(sigma12_em));
pdf_2_em = makedist('Normal', mu2_em, sqrt(sigma22_em));

y1_em = w1_em*pdf(pdf_1_mle, x);
y2_em = w2_em*pdf(pdf_2_mle, x);
y_mixed_em = y1_em + y2_em;

p1_ = plot(x, y1_em, 'LineWidth', 2);
p2_ = plot(x, y2_em, 'LineWidth', 2);
p3_ = plot(x, y_mixed_em, 'LineWidth', 2);

hold off


% 11.-Extend the codes to the case of bivariate Gaussian till the next
% tutorial
