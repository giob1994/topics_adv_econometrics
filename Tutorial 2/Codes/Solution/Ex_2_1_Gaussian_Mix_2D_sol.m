% The objective of this code is simulating out of a 2-dimensional Gaussian mixture
% model and fitting a Gaussian mixture distribution to a
% data set by maximizing the log-likelihood, using Expectation-Maximization
% (EM)  algorithm and built-in matlab function with implemented EM

% We extend here the code to the case of mixture of two-dimensional 
% Gaussian distributions

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Start with number of components n_components = 2;
% Variables:
% n sample size
% mu1, mu2 means of the Gaussians used in the mixture
% sigma12, sigma22 variances of the Gaussians used in the mixture
% w1, w2 weights for each component of the mixture

clear
close all
rng default;  % For reproducibility
n_components = 2;
% dimensionality of X
d = 2; 
p_mixture_1 = 1/3;

% set weights
w = zeros(n_components, 1);
w(1) = p_mixture_1;
w(2) = 1 - w(1);

% parameters for the first component (class Y = 1)
% mean
mu_1(1) = -5;
mu_1(2) = -1;
% covariance
sigma2_x1_1 = 30;
sigma2_x2_1 = 10;
sigma_x1x2_1 = 5;
Sigma_1 = [sigma2_x1_1, sigma_x1x2_1; sigma_x1x2_1, sigma2_x2_1];
if min(eigs(Sigma_1)) <= 1.e-5
    error('Covariance matrix Sigma_1 is close to or is not PSD');
end
% parameters for the first component (class Y = 0)
% mean
mu_0(1) = 10;
mu_0(2) = 5;
% covariance
sigma2_x1_0 = 40;
sigma2_x2_0 = 12;
sigma_x1x2_0 = 10;
Sigma_0 = [sigma2_x1_0, sigma_x1x2_0; sigma_x1x2_0, sigma2_x2_0];
if min(eigs(Sigma_0)) <= 1.e-5
    error('Covariance matrix Sigma_0 is close to or is not PSD');
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% 1.-Generate a Gaussian mixture distributed sample of size n of means mu1, mu2,
% variances sigma12 and sigma22, and mixing weights w1, w2
% (a) using no built-in special functions
% using Matlab functions: (b) binornd, (c) gmdistribution

% we need to simulate 
% {X|Y=1}~N(mu_1, Sigma_1)
% {X|Y=0}~N(mu_0, Sigma_0)

% sample length
n = 1000;
% (a) no built in functions
Indexes = rand(n, 1);
Gaussian_sample1 = mvnrnd(mu_1, Sigma_1, n);
Gaussian_sample2 = mvnrnd(mu_0, Sigma_0, n);
Gaussian_sample1_man = Gaussian_sample1(Indexes <= p_mixture_1);
Gaussian_sample2_man = Gaussian_sample2(Indexes > p_mixture_1);
n_component1 = length(Gaussian_sample1);
n_component2 = length(Gaussian_sample2);
GaussMix_sample_man = [Gaussian_sample1_man; Gaussian_sample2_man];

% (b) using Matlab functions: binornd
Y = binornd(1, p_mixture_1, n, 1);
Gaussian_sample1 = mvnrnd(mu_1, Sigma_1, n);
Gaussian_sample2 = mvnrnd(mu_0, Sigma_0, n);
Gaussian_sample1_brnll = Gaussian_sample1(Y == 1);
Gaussian_sample2_brnll = Gaussian_sample2(Y == 0);
n_component1 = sum(Y);
n_component2 = n - n_component1;
GaussMix_sample_brnll = [Gaussian_sample1_brnll; Gaussian_sample2_brnll];

% (b) using Matlab functions: gmdistribution
% no labels are available 
mu = [mu_1; mu_0];
sigma = cat(3, Sigma_1, Sigma_0);
p = [p_mixture_1, 1 - p_mixture_1];
obj = gmdistribution(mu, sigma, p);
GaussMix_sample_gmm = random(obj, n);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% 2.- (a) Create a 2D scatter plot that represents the simulated data. 
% Denote the observations from the class Y=1 with the symbol '+'
% and with 'o' the observations from the class Y=0

figure(1)
plot(Gaussian_sample1(:,1),Gaussian_sample1(:,2),'+b');
hold on;
plot(Gaussian_sample2(:,1),Gaussian_sample2(:,2),'or');
xlabel('First feature')
ylabel('Second feature')
title('Scatter plot with the observations from the class Y=1 (+) and from the class Y=0 (o)')
legend('class Y=1','class Y=0','Location','northwest')
hold off;

% 2.- (b) Plot weighted
% versions of the two Gaussians used in the simulation, and the Gaussian
% mixture distribution

figure(2)
l1 = min(GaussMix_sample_gmm(:, 1));
u1 = max(GaussMix_sample_gmm(:, 1));
l2 = min(GaussMix_sample_gmm(:, 2));
u2 = max(GaussMix_sample_gmm(:, 2));
x1 = linspace(l1, u1, 100);
x2 = linspace(l2, u2, 100);
x = [x1; x2];
[X1, X2] = meshgrid(x1, x2);
% Plot the first Gaussian weighted
y1 = w(1) * mvnpdf([X1(:), X2(:)], mu_1, Sigma_1);
y1_plot = reshape(y1, length(x1), length(x2));
surf(x1, x2, y1_plot);
xlabel('Feature 1'); ylabel('Feature 2'); zlabel('Probability Density');
hold on;
% Plot the second Gaussian weighted
y2 = w(2) * mvnpdf([X1(:), X2(:)], mu_0, Sigma_0);
y2_plot = reshape(y2, length(x1), length(x2));
surf(x1, x2, y2_plot);
% Plot the Gaussian Mixture
y = y1_plot + y2_plot;
surf(x1, x2, y);
hold off

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% 3.-Create a function getGaussianMixLogLikelihood(x, theta) that evaluates the
% Gaussian Mixture Log-likelihood for any number of components at a sample x 
% with the vector theta of concatenated means mu, variances sigma2, and weights w
% verify that it works the same as the previous function provided that the
% number of components is 2
mu_0 = mu_0';
mu_1 = mu_1';

% take vech(Sigma_1)
Sigma_1_vals = [Sigma_1(1,1); Sigma_1(2,1); Sigma_1(2,2)];
% take vech(Sigma_0)
Sigma_0_vals = [Sigma_0(1,1); Sigma_0(2,1); Sigma_0(2,2)];
%concatenate
theta = [mu_1; mu_0; Sigma_1_vals; Sigma_0_vals; w];
loglikVal = getGaussianMixLogLikelihood(GaussMix_sample_gmm, theta);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% 4.-Fit a Gaussian Mixture to the simulated sample by maximizing the log-likelihood
% Enforce in the maximization process the PSD of covariances and
% that the sum of weights equals one

funToMinimize = @(theta) -getGaussianMixLogLikelihood(GaussMix_sample_gmm, theta);
options = optimoptions(@fmincon,'Display','iter', 'TolX', 1.e-4);
Aeq = zeros(1, length(theta));
Aeq(end - 1) = 1;
Aeq(end) = 1;
beq = 1;
Alb = [-Inf -Inf -Inf -Inf 0 0 0 0 0 0 0 0];
Blf = [Inf Inf Inf Inf Inf Inf Inf Inf Inf Inf 1 1];
theta0 = [0 0 0 0 Sigma_1_vals' Sigma_0_vals' 0.5 0.5];
[theta_mle, loglike1] = fmincon(funToMinimize, theta0,...
    [], [], Aeq, beq, Alb, Blf, @confuneq, options);
% find which component corresponds to the first simulated by looking at the
% values of means
[~, ind] = min([theta_mle(1), theta_mle(3)]);
if (ind == 1)
    mu_1_mle = [theta_mle(1); theta_mle(2)];
    mu_0_mle = [theta_mle(3); theta_mle(4)];
    Sigma_1_mle = [theta_mle(5), theta_mle(6); theta_mle(6), theta_mle(7)];
    Sigma_0_mle = [theta_mle(8), theta_mle(9); theta_mle(9), theta_mle(10)];
    w_mle = [theta_mle(end-1); theta_mle(end)];
else
    mu_0_mle = [theta_mle(1); theta_mle(2)];
    mu_1_mle = [theta_mle(3); theta_mle(4)];
    Sigma_0_mle = [theta_mle(5), theta_mle(6); theta_mle(6), theta_mle(7)];
    Sigma_1_mle = [theta_mle(8), theta_mle(9); theta_mle(9), theta_mle(10)];
    w_mle = [theta_mle(end); theta_mle(end-1)];
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% 5.-Use matlab function fitgmdist to fit a Gaussian Mixture to the simulated sample

rng(1); % Reset seed for common start values
% fit Gaussian mixture
Options = statset('Display', 'iter', 'MaxIter', 1500, 'TolFun', 1e-4);
n_components = 2;
try
    GMModel = fitgmdist(GaussMix_sample_gmm, n_components, 'Options', Options, 'RegularizationValue',0.0000001);
catch exception
    disp('There was an error fitting the Gaussian mixture model');
    error = exception.message;
end
% find which component corresponds to the first simulated by looking at the
% values of means
mu_temp1 = GMModel.mu(1, 1);
mu_temp2 = GMModel.mu(2, 1);

[~, ind] = min([mu_temp1, mu_temp2]);
if (ind == 1)
    mu_1_em = GMModel.mu(1, :);
    mu_0_em = GMModel.mu(2, :);
    Sigma_1_em = GMModel.Sigma(:, :, 1);
    Sigma_0_em = GMModel.Sigma(:, :, 2);
    w_em = [theta_mle(end-1); theta_mle(end)];
else
    mu_0_em = GMModel.mu(1, :);
    mu_1_em = GMModel.mu(2, :);
    Sigma_0_em = GMModel.Sigma(:, :, 1);
    Sigma_1_em = GMModel.Sigma(:, :, 2);
    w_em = [theta_mle(end); theta_mle(end-1)];
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% 6.-Use your own EM algorithm to estimate the Gaussian mixture model. EM
% is used in fitgmdist but the goal is to implement it on our own
k = 2;

indeces = randperm(n);
mu = zeros(d, k);
mu_current = zeros(d, k);
Sigma = struct('val', zeros(d));
for i = 1:k
    mu(:, i) = GaussMix_sample_gmm(indeces(i), i);
    Sigma(i).val = cov(GaussMix_sample_gmm);
end
% Use the total sample variance of the dataset as the initial variance
% value
% Assign equal prior probabilities to each class
p_class = ones(1, k) * (1 / k);

% save probability P(Y=k|X=x_i) for each class
W = zeros(n, k);
TolX = 1.e-6;
% run EM
iter_num = 0;
while (norm(mu - mu_current, 2) > TolX)
    iter_num = iter_num + 1;
    fprintf('  EM Iteration %d\n', iter_num);

    %===============================================
    % STEP 1: Expectation
    
    pdf_values = zeros(n, k);
    
    for j = 1:k
        % Evaluate the Gaussian for all data points for each class
        pdf_values(:, j) = mvnpdf(GaussMix_sample_gmm, mu(:, j)', Sigma(j).val);
    end
    
    % Multiply each pdf value by the prior probability for each class
    pdf_w = pdf_values .* repmat(p_class, n, 1);
    
   % Divide the weighted probabilities by the sum of weighted
    % probabilities for each class
    W = pdf_w./repmat(sum(pdf_w, 2), 1, 2);
    
    %===============================================
    % STEP 2: Maximization

    mu_current = mu;    
    
    for j = 1:k
    
        % Calculate the prior probability for each class
        p_class(j) = mean(W(:, j));
        
        % Calculate the new mean for each class by taking the weighted
        % average of *all* data points.
        temp = W(:, j)' * GaussMix_sample_gmm;
        mu(:,j) = temp./sum(W(:, j), 1);
    
        % Calculate the variance for each class by taking the weighted
        % average of the squared differences from the mean for all data
        % points
        valSigma = zeros(d);
        for s = 1:n
            valSigma = valSigma + W(s, j) * (GaussMix_sample_gmm(s,:)' - mu(:,j))*(GaussMix_sample_gmm(s,:) - mu(:,j)');
        end
        Sigma(j).val = valSigma./sum(W(:, j), 1);
    end
end

% find which component corresponds to the first simulated by looking at the
% values of means
mu_temp1 = mu(1, 1);
mu_temp2 = mu(1, 2);

[~, ind] = min([mu_temp1, mu_temp2]);
if (ind == 1)
    mu_1_em_stud = mu(:, 1);
    mu_0_em_stud = mu(:, 2);
    Sigma_1_em_stud = Sigma(1).val;
    Sigma_0_em_stud = Sigma(2).val;
    w_em_stud = [theta_mle(end-1); theta_mle(end)];
else
    mu_1_em_stud = mu(:, 2);
    mu_0_em_stud = mu(:, 1);
    Sigma_1_em_stud = Sigma(2).val;
    Sigma_0_em_stud = Sigma(1).val;
    w_em_stud = [theta_mle(end); theta_mle(end-1)];
end
