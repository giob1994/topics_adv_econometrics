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

fig_id = 1;

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

rng(100);

Bern = binornd(1, p_mixture, n, 1);

mu1 = [mu_x1_1; mu_x2_1];
Sigma1 = [sigma2_x1_1, sigma_x1x2_1; sigma_x1x2_1, sigma2_x2_1];
mu0 = [mu_x1_0; mu_x2_0];
Sigma0 = [sigma2_x1_0, sigma_x1x2_0; sigma_x1x2_0, sigma2_x2_0];

ILL_subsample = mvnrnd(mu1, Sigma1, n);
ILL_subsample = ILL_subsample(Bern == 1, :);
HEALTHY_subsample = mvnrnd(mu0, Sigma0, n);
HEALTHY_subsample = HEALTHY_subsample(Bern == 0, :);
MIXED_sample = [ILL_subsample; HEALTHY_subsample];
MIXED_class = [ones(size(ILL_subsample, 1), 1); zeros(size(HEALTHY_subsample, 1), 1)];

n_component1 = size(ILL_subsample, 1);
n_component0 = size(HEALTHY_subsample, 1);
                    
lim_x = [min(MIXED_sample(:,1))-10, max(MIXED_sample(:,1))+10];
lim_y = [min(MIXED_sample(:,2))-10, max(MIXED_sample(:,2))+10];

%% 2.- Create a 2D scatter plot that represents the hemoglobine level and health of the
% anemic and healthy subjects. Denote the anemic subjects with the symbol '+'
% and with 'o' the healthy ones

figure(fig_id)
fig_id = fig_id+1;
subplot(2,1,1)
hold on
title('Separate samples')
plot(ILL_subsample(:,1), ILL_subsample(:,2), 'b+')
plot(HEALTHY_subsample(:,1), HEALTHY_subsample(:,2), 'ro')
grid on
legend('ILL sample', 'HEALTHY sample')
hold off
subplot(2,1,2)
hold on
title('Mixed sample')
plot(MIXED_sample(:,1), MIXED_sample(:,2), 'kx')
grid on
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

bayes_ill = MIXED_sample(eta_f(MIXED_sample) > 0, :);
bayes_healthy = MIXED_sample(eta_f(MIXED_sample) <= 0, :);

% Errors - Type 1 and Type 2:
error_Typ1_bayes = setdiff(bayes_ill, ILL_subsample, 'rows');
error_Typ2_bayes = setdiff(bayes_healthy, HEALTHY_subsample, 'rows');
bayes_emp_risk = (length(error_Typ1_bayes) + length(error_Typ2_bayes))/n;

disp(' ')
disp(' Bayes empirical risk:')
disp(' ')
disp([' r = ', num2str(bayes_emp_risk)])
disp(' ')

%% 4.- Redo the plot in point 2 and mark in red the markers corresponding to the 
% subjects that are misclassified by the Bayes classifier

figure(fig_id)
fig_id = fig_id+1;

hold on
title('Bayes classifier error')
plot(ILL_subsample(:,1), ILL_subsample(:,2), 'k+')
plot(HEALTHY_subsample(:,1), HEALTHY_subsample(:,2), 'ko')
plot(error_Typ1_bayes(:,1), error_Typ1_bayes(:,2), 'rx', 'MarkerSize', 12)
plot(error_Typ2_bayes(:,1), error_Typ2_bayes(:,2), 'gx', 'MarkerSize', 12)
xlim(lim_x)
ylim(lim_y)
bound_image = getDecisionBoundaryPlot(lim_x, lim_y, eta_f, 1);
colormap parula
imagesc(lim_x, lim_y, bound_image, 'AlphaData', 0.2)
grid on
legend('ILL sample', 'HEALTHY sample', 'Type 1 errors', 'Type 2 errors', ...
            'Location', 'best')
hold off

%% 5.- Construct the logistic classifier based on the training sample and
% compute the associated errors and classifier's empirical risk.

logit_crit = @(x, beta) exp([ones(size(x,1),1), x]*beta(:));

log_beta = glmfit(MIXED_sample, MIXED_class, 'binomial', 'link', 'logit');

logit_ill = MIXED_sample(logit_crit(MIXED_sample, log_beta) > 1, :);
logit_healthy = MIXED_sample(logit_crit(MIXED_sample, log_beta) <= 1, :);

% Errors - Type 1 and Type 2:
error_Typ1_logit = setdiff(logit_ill, ILL_subsample, 'rows');
error_Typ2_logit = setdiff(logit_healthy, HEALTHY_subsample, 'rows');

% Empirical risk:
logit_emp_risk = (length(error_Typ1_logit) + length(error_Typ2_logit))/n;

disp(' ')
disp(' Logit empirical risk:')
disp(' ')
disp([' r = ', num2str(logit_emp_risk)])
disp(' ')

%% 6.- Redo the plot in point 2 and mark in red the markers corresponding to the 
% subjects that are misclassified by the logistic classifier.

figure(fig_id)
fig_id = fig_id+1;
hold on
title('Logistic classifier error')
plot(ILL_subsample(:,1), ILL_subsample(:,2), 'k+')
plot(HEALTHY_subsample(:,1), HEALTHY_subsample(:,2), 'ko')
plot(error_Typ1_logit(:,1), error_Typ1_logit(:,2), 'rx', 'MarkerSize', 12)
plot(error_Typ2_logit(:,1), error_Typ2_logit(:,2), 'gx', 'MarkerSize', 12)
xlim(lim_x)
ylim(lim_y)
bound_image = getDecisionBoundaryPlot(lim_x, lim_y, ...
                                    @(x) logit_crit(x, log_beta)-1, 1);
colormap parula
imagesc(lim_x, lim_y, bound_image, 'AlphaData', 0.2)
grid on
legend('ILL sample', 'HEALTHY sample', 'Type 1 errors', 'Type 2 errors', ...
            'Location', 'best')
hold off

%% 7.- Construct the LDA classifier based on the training sample and 
% compute the associated errors and classifier's empirical risk.

w1_lda      = n_component1/n;
w0_lda      = n_component0/n;
mu1_lda     = mean(ILL_subsample);
mu0_lda     = mean(HEALTHY_subsample);
Sigma12_s   = cov(ILL_subsample - mu1_lda);
Sigma02_s   = cov(HEALTHY_subsample - mu0_lda);
Sigma_lda   = (1/(n-2))*(Sigma12_s+Sigma02_s);
                            
delta_r_lda   = @(X, w, mu, Sigma) X*(Sigma\mu')-1/2*(mu*(Sigma\mu'))+log(w);
delta_ill     = delta_r_lda(MIXED_sample, w1_lda, mu1_lda, Sigma_lda);
delta_healthy = delta_r_lda(MIXED_sample, w0_lda, mu0_lda, Sigma_lda);

LDA_ill = MIXED_sample(delta_ill > delta_healthy, :);
LDA_healthy = MIXED_sample(delta_ill <= delta_healthy, :);

% Errors - Type 1 and Type 2:
error_Typ1_LDA = setdiff(LDA_ill, ILL_subsample, 'rows');
error_Typ2_LDA = setdiff(LDA_healthy, HEALTHY_subsample, 'rows');
LDA_emp_risk = (length(error_Typ1_LDA) + length(error_Typ2_LDA))/n;

disp(' ')
disp(' LDA empirical risk:')
disp(' ')
disp([' r = ', num2str(LDA_emp_risk)])
disp(' ')

% Plot:
figure(fig_id)
fig_id = fig_id+1;
hold on
title('LDA classifier error')
plot(ILL_subsample(:,1), ILL_subsample(:,2), 'k+')
plot(HEALTHY_subsample(:,1), HEALTHY_subsample(:,2), 'ko')
plot(error_Typ1_LDA(:,1), error_Typ1_LDA(:,2), 'rx', 'MarkerSize', 12)
plot(error_Typ2_LDA(:,1), error_Typ2_LDA(:,2), 'gx', 'MarkerSize', 12)
xlim(lim_x)
ylim(lim_y)
bound_image = getDecisionBoundaryPlot(lim_x, lim_y, ...
              @(x) delta_r_lda(x, w1_lda, mu1_lda, Sigma_lda) - ...
                   delta_r_lda(x, w0_lda, mu0_lda, Sigma_lda), 1);
colormap parula
imagesc(lim_x, lim_y, bound_image, 'AlphaData', 0.2)
grid on
legend('ILL sample', 'HEALTHY sample', 'Type 1 errors', 'Type 2 errors', ...
            'Location', 'best')
hold off


%% 8.- Construct the QDA classifier based on the training sample and
% compute the associated errors and classifier's empirical risk.

w1_qda = n_component1/n;
w0_qda = n_component0/n;
mu1_qda = mean(ILL_subsample);
mu0_qda = mean(HEALTHY_subsample);
Sigma12_qda = cov(ILL_subsample - mu1_qda);
Sigma02_qda = cov(HEALTHY_subsample - mu0_qda);
                            
delta_r_qda = @(X, w, mu, Sigma) - 1/2*log(det(Sigma)) ...
                                 - 1/2*sum((X-mu).*(Sigma\(X-mu)')', 2) ...
                                 + log(w) ;

delta_ill = delta_r_qda(MIXED_sample, w1_qda, mu1_qda, Sigma12_qda);
delta_healthy = delta_r_qda(MIXED_sample, w0_qda, mu0_qda, Sigma02_qda);

QDA_ill = MIXED_sample(delta_ill > delta_healthy, :);
QDA_healthy = MIXED_sample(delta_ill <= delta_healthy, :);

% Errors - Type 1 and Type 2:
error_Typ1_QDA = setdiff(QDA_ill, ILL_subsample, 'rows');
error_Typ2_QDA = setdiff(QDA_healthy, HEALTHY_subsample, 'rows');
QDA_emp_risk = (length(error_Typ1_QDA) + length(error_Typ2_QDA))/n;

disp(' ')
disp(' QDA empirical risk:')
disp(' ')
disp([' r = ', num2str(QDA_emp_risk)])
disp(' ')

% Plot:
figure(fig_id)
fig_id = fig_id+1;
hold on
title('QDA classifier error')
plot(ILL_subsample(:,1), ILL_subsample(:,2), 'k+')
plot(HEALTHY_subsample(:,1), HEALTHY_subsample(:,2), 'ko')
plot(error_Typ1_QDA(:,1), error_Typ1_QDA(:,2), 'rx', 'MarkerSize', 12)
plot(error_Typ2_QDA(:,1), error_Typ2_QDA(:,2), 'gx', 'MarkerSize', 12)
xlim(lim_x)
ylim(lim_y)
bound_image = getDecisionBoundaryPlot(lim_x, lim_y, ...
              @(x) delta_r_qda(x, w1_qda, mu1_qda, Sigma12_qda) - ...
                   delta_r_qda(x, w0_qda, mu0_qda, Sigma02_qda), 1);
colormap parula
imagesc(lim_x, lim_y, bound_image, 'AlphaData', 0.2)
grid on
legend('ILL sample', 'HEALTHY sample', 'Type 1 errors', 'Type 2 errors', ...
            'Location', 'best')
hold off


%% 7.- Construct the Naive Bayes classifier based on the training sample and 
% compute the associated errors and classifier's empirical risk.

w1_nbc = n_component1/n;
w0_nbc = n_component0/n;
mu1_nbc = mean(ILL_subsample);
mu0_nbc = mean(HEALTHY_subsample);
Sigma12_nbc = diag(cov(ILL_subsample - mu1_nbc));
Sigma02_nbc = diag(cov(HEALTHY_subsample - mu0_nbc));

delta_r_nbc = @(X) ...
     w1_nbc*normpdf(X(:,1), mu1_nbc(1), sqrt(Sigma12_nbc(1))).* ... 
            normpdf(X(:,2), mu1_nbc(2), sqrt(Sigma12_nbc(2))) ...
 ./ (w1_nbc*normpdf(X(:,1), mu1_nbc(1), sqrt(Sigma12_nbc(1))).* ...
            normpdf(X(:,2), mu1_nbc(2), sqrt(Sigma12_nbc(2))) ... 
   + w0_nbc*normpdf(X(:,1), mu0_nbc(1), sqrt(Sigma02_nbc(1))).* ...
            normpdf(X(:,2), mu0_nbc(2), sqrt(Sigma02_nbc(2))));

delta_ill = delta_r_nbc(MIXED_sample);
delta_healthy = 1 - delta_ill;

NBC_ill = MIXED_sample(delta_ill > delta_healthy, :);
NBC_healthy = MIXED_sample(delta_ill <= delta_healthy, :);

% Errors - Type 1 and Type 2:
error_Typ1_NBC = setdiff(NBC_ill, ILL_subsample, 'rows');
error_Typ2_NBC = setdiff(NBC_healthy, HEALTHY_subsample, 'rows');
NBC_emp_risk = (length(error_Typ1_NBC) + length(error_Typ2_NBC))/n;

disp(' ')
disp(' Naive Bayes classifier empirical risk:')
disp(' ')
disp([' r = ', num2str(NBC_emp_risk)])
disp(' ')

% Plot:
figure(fig_id)
fig_id = fig_id+1;
hold on
title('Naive Bayes classifier error')
plot(ILL_subsample(:,1), ILL_subsample(:,2), 'k+')
plot(HEALTHY_subsample(:,1), HEALTHY_subsample(:,2), 'ko')
plot(error_Typ1_NBC(:,1), error_Typ1_NBC(:,2), 'rx', 'MarkerSize', 12)
plot(error_Typ2_NBC(:,1), error_Typ2_NBC(:,2), 'gx', 'MarkerSize', 12)
xlim(lim_x)
ylim(lim_y)
bound_image = getDecisionBoundaryPlot(lim_x, lim_y, ...
              @(x) delta_r_nbc(x)-1/2, 1);
colormap parula
imagesc(lim_x, lim_y, bound_image, 'AlphaData', 0.2)
grid on
legend('ILL sample', 'HEALTHY sample', 'Type 1 errors', 'Type 2 errors', ...
            'Location', 'best')
hold off

%% 9.- Construct perceptron hyperplane classifier based on the training sample and 
% compute the classifier's empirical risk.
% The perceptron is specified by a vector beta = [beta0, beta1, beta2] that 
% determines a separating hyperplane beta0 + beta1 * x1 + beta2 * x2
% Start by constructing a function that computes the empirical error of this
% classifier as a function of beta.

h_sample = [ones(size(MIXED_sample,1),1), MIXED_sample];

percept_emp_error = @(beta) sum((h_sample*beta(:) < 0) == MIXED_class)/n;
beta_percept = ga(percept_emp_error, 3);
percept_ill = MIXED_sample((h_sample*beta_percept(:) <= 0), :);
percept_healthy = MIXED_sample((h_sample*beta_percept(:) > 0), :);

% Type 1 errors:
error_Typ1_percept = setdiff(percept_ill, ILL_subsample, 'rows');
error_Typ2_percept = setdiff(percept_healthy, HEALTHY_subsample, 'rows');
percept_emp_risk = (length(error_Typ1_percept) + length(error_Typ2_percept))/n;

disp(' ')
disp(' Perceptron classifier empirical risk:')
disp(' ')
disp([' r = ', num2str(percept_emp_risk)])
disp(' ')

% Plot:
figure(fig_id)
fig_id = fig_id+1;
hold on
title('Perceptron classifier error')
plot(ILL_subsample(:,1), ILL_subsample(:,2), 'k+')
plot(HEALTHY_subsample(:,1), HEALTHY_subsample(:,2), 'ko')
plot(error_Typ1_percept(:,1), error_Typ1_percept(:,2), 'rx', 'MarkerSize', 12)
plot(error_Typ2_percept(:,1), error_Typ2_percept(:,2), 'gx', 'MarkerSize', 12)
xlim(lim_x)
ylim(lim_y)
bound_image = getDecisionBoundaryPlot(lim_x, lim_y, ...
              @(x) [ones(size(x,1),1), x]*beta_percept(:) < 0, 1);
colormap parula
imagesc(lim_x, lim_y, bound_image, 'AlphaData', 0.2)
grid on
legend('ILL sample', 'HEALTHY sample', 'Type 1 errors', 'Type 2 errors', ...
            'Location', 'best')
hold off

%% 10.- Classify the data with a KNN classifier and compute the associated errors 
% and classifier's empirical risk with 1, 5, and 10 neighbors

knn_1 = fitcknn(MIXED_sample, MIXED_class ,'NumNeighbors', 1);
knn_1_class = predict(knn_1, MIXED_sample);

% Errors - Type 1 and Type 2:
error_Typ1_knn_1 = logical((knn_1_class ~= MIXED_class).*MIXED_class);
error_Typ2_knn_1 = logical(((1-knn_1_class) ~= (1-MIXED_class)).*(1-MIXED_class));
knn_1_emp_risk = (sum(error_Typ1_knn_1) + sum(error_Typ2_knn_1))/n;

disp(' ')
disp(' K-Nearest N. (1 neighbour) empirical risk:')
disp(' ')
disp([' r = ', num2str(knn_1_emp_risk)])
disp(' ')

% Plot:
figure(fig_id)
fig_id = fig_id+1;
hold on
title('K-Nearest Neighbors classifier error - 1 Neighbor')
plot(ILL_subsample(:,1), ILL_subsample(:,2), 'k+')
plot(HEALTHY_subsample(:,1), HEALTHY_subsample(:,2), 'ko')
plot(MIXED_sample(error_Typ1_knn_1,1), MIXED_sample(error_Typ1_knn_1,2), 'rx', 'MarkerSize', 12)
plot(MIXED_sample(error_Typ2_knn_1,1), MIXED_sample(error_Typ2_knn_1,2), 'gx', 'MarkerSize', 12)
xlim(lim_x)
ylim(lim_y)
bound_image = getDecisionBoundaryPlot(lim_x, lim_y, ...
              @(x) predict(knn_1, x), 1);
colormap parula
imagesc(lim_x, lim_y, bound_image, 'AlphaData', 0.2)
grid on
legend('ILL sample', 'HEALTHY sample', 'Type 1 errors', 'Type 2 errors', ...
            'Location', 'best')
hold off


knn_5 = fitcknn(MIXED_sample, MIXED_class ,'NumNeighbors', 5);
knn_5_class = predict(knn_5, MIXED_sample);

% Errors - Type 1 and Type 2:
error_Typ1_knn_5 = logical((knn_5_class ~= MIXED_class).*MIXED_class);
error_Typ2_knn_5 = logical(((1-knn_5_class) ~= (1-MIXED_class)).*(1-MIXED_class));
knn_5_emp_risk = (sum(error_Typ1_knn_5) + sum(error_Typ2_knn_5))/n;

disp(' ')
disp(' K-Nearest N. (5 neighbours) empirical risk:')
disp(' ')
disp([' r = ', num2str(knn_5_emp_risk)])
disp(' ')

% Plot:
figure(fig_id)
fig_id = fig_id+1;

hold on
title('K-Nearest Neighbors classifier error - 5 Neighbor')
plot(ILL_subsample(:,1), ILL_subsample(:,2), 'k+')
plot(HEALTHY_subsample(:,1), HEALTHY_subsample(:,2), 'ko')
plot(MIXED_sample(error_Typ1_knn_5,1), MIXED_sample(error_Typ1_knn_5,2), 'rx', 'MarkerSize', 12)
plot(MIXED_sample(error_Typ2_knn_5,1), MIXED_sample(error_Typ2_knn_5,2), 'gx', 'MarkerSize', 12)
xlim(lim_x)
ylim(lim_y)
bound_image = getDecisionBoundaryPlot(lim_x, lim_y, ...
              @(x) predict(knn_5, x), 1);
colormap parula
imagesc(lim_x, lim_y, bound_image, 'AlphaData', 0.2)
grid on
legend('ILL sample', 'HEALTHY sample', 'Type 1 errors', 'Type 2 errors', ...
            'Location', 'best')
hold off

knn_10 = fitcknn(MIXED_sample, MIXED_class ,'NumNeighbors', 10);
knn_10_class = predict(knn_10, MIXED_sample);

% Errors - Type 1 and Type 2:
error_Typ1_knn_10 = logical((knn_10_class ~= MIXED_class).*MIXED_class);
error_Typ2_knn_10 = logical((knn_10_class ~= MIXED_class).*(1-MIXED_class));
knn_10_emp_risk = (sum(error_Typ1_knn_10) + sum(error_Typ2_knn_10))/n;

disp(' ')
disp(' K-Nearest N. (10 neighbours) empirical risk:')
disp(' ')
disp([' r = ', num2str(knn_10_emp_risk)])
disp(' ')

% Plot:
figure(fig_id)
fig_id = fig_id+1;
hold on
title('K-Nearest Neighbors classifier error - 10 Neighbor')
plot(ILL_subsample(:,1), ILL_subsample(:,2), 'k+')
plot(HEALTHY_subsample(:,1), HEALTHY_subsample(:,2), 'ko')
plot(MIXED_sample(error_Typ1_knn_10,1), MIXED_sample(error_Typ1_knn_10,2), 'rx', 'MarkerSize', 12)
plot(MIXED_sample(error_Typ2_knn_10,1), MIXED_sample(error_Typ2_knn_10,2), 'gx', 'MarkerSize', 12)
xlim(lim_x)
ylim(lim_y)
bound_image = getDecisionBoundaryPlot(lim_x, lim_y, ...
              @(x) predict(knn_10, x), 1);
colormap parula
imagesc(lim_x, lim_y, bound_image, 'AlphaData', 0.2)
grid on
legend('ILL sample', 'HEALTHY sample', 'Type 1 errors', 'Type 2 errors', ...
            'Location', 'best')
hold off

%% 11.- Classify the data using the unsupervised approach providing the number of classes 2
% (for that make use of the EM and MLE techniques implemented earlier; for the MLE 
% technique make an improvement and guarantee PSD constraint by construction using the Cholesky
% decomposition of covariance matrix), compute the associated errors and classifier's empirical risk.

% MLE estimation:

dims = 2;

loglikeF = @(theta) -getBiGaussianMixLogLikelihood(MIXED_sample, theta, dims);
options = optimoptions(@fmincon, 'Display', 'none');

x0 = [1/2, 1/2, ...
      160, 60,  100, 30, ...
      400, 100, 300,  500, 200, 300];
Aeq = [1,  1,  0, 0,  0, 0,  0, 0, 0,  0, 0, 0];
beq = 1;
lb = [0,  0,  -Inf, -Inf,  -Inf, -Inf,  0, 0, 0,  0, 0, 0];
ub = [1,  1,   Inf,  Inf,   Inf,  Inf,  Inf, Inf, Inf,  Inf, Inf, Inf];

[theta_mle, log_like_mle] = ...
    fmincon(loglikeF, x0, [], [], Aeq, beq, lb, ub, @defposconst, options);

w0_mle      = theta_mle(1);
w1_mle      = theta_mle(2);
mu0_mle     = theta_mle(3:2+dims);
mu1_mle     = theta_mle(3+dims:2+2*dims);
Sigma0_mle  = [ theta_mle(3+2*dims), theta_mle(4+2*dims);
                theta_mle(4+2*dims), theta_mle(5+2*dims); ];
Sigma1_mle  = [ theta_mle(3+2*dims+dims^2-1), theta_mle(4+2*dims+dims^2-1);
                theta_mle(4+2*dims+dims^2-1), theta_mle(5+2*dims+dims^2-1); ];
            
pdf_0_mle = @(X) w0_mle*(mvnpdf(X, mu0_mle, Sigma0_mle));
pdf_1_mle = @(X) w1_mle*(mvnpdf(X, mu1_mle, Sigma1_mle));

eta_mle = @(X) (pdf_0_mle(X))./(pdf_0_mle(X) + pdf_1_mle(X));
MLE_class = (eta_mle(MIXED_sample) <= 1/2);
                    
% Errors - Type 1 and Type 2:
error_Typ1_MLE = logical((MLE_class ~= MIXED_class).*MIXED_class);
error_Typ2_MLE = logical((MLE_class ~= MIXED_class).*(1-MIXED_class));
MLE_emp_risk = (sum(error_Typ1_MLE) + sum(error_Typ2_MLE))/n;

disp(' ')
disp(' MLE empirical risk:')
disp(' ')
disp([' r = ', num2str(MLE_emp_risk)])
disp(' ')

% Plot:
figure(fig_id)
fig_id = fig_id+1;
hold on
title('MLE classifier error')
plot(ILL_subsample(:,1), ILL_subsample(:,2), 'k+')
plot(HEALTHY_subsample(:,1), HEALTHY_subsample(:,2), 'ko')
plot(MIXED_sample(error_Typ1_MLE,1), MIXED_sample(error_Typ1_MLE,2), 'rx', 'MarkerSize', 12)
plot(MIXED_sample(error_Typ2_MLE,1), MIXED_sample(error_Typ2_MLE,2), 'gx', 'MarkerSize', 12)
xlim(lim_x)
ylim(lim_y)
bound_image = getDecisionBoundaryPlot(lim_x, lim_y, ...
              @(x) eta_mle(x)-1/2, 1);
colormap parula
imagesc(lim_x, lim_y, bound_image, 'AlphaData', 0.2)
grid on
legend('ILL sample', 'HEALTHY sample', 'Type 1 errors', 'Type 2 errors', ...
            'Location', 'best')
hold off
            

% PDF 3D plot:
x1 = linspace(lim_x(1), lim_x(2), 100);
x2 = linspace(lim_y(1), lim_y(2), 100);
[X1, X2] = meshgrid(x1, x2);
pdf_mixed = reshape(pdf_0_mle([X1(:), X2(:)]) + pdf_1_mle([X1(:), X2(:)]), 100, 100);
figure(fig_id)
fig_id = fig_id+1;
hold on
title('MLE density estimation')
plot(ILL_subsample(:,1), ILL_subsample(:,2), 'b+')
plot(HEALTHY_subsample(:,1), HEALTHY_subsample(:,2), 'ro')                
surf(x1, x2, pdf_mixed, 'FaceAlpha', 0.5);
xlim(lim_x)
ylim(lim_y)
shading interp
colormap jet
pbaspect([1 1 0.5])
view(3)
grid on
legend('ILL sample', 'HEALTHY sample', 'MLE mixed density', ...
            'Location', 'best')
        
        
% EM estimation:

precision = 10^-5;

theta_0 = [2, ...
           160, 60,  100, 30, ...
           400, 100, 100, 300,  500, 200, 200, 300, ...
           1/2, 1/2];

theta_em = fitMultiMixedGaussianEM(MIXED_sample, theta_0, precision);

mu0_em = theta_em.mu(1,:);
mu1_em = theta_em.mu(2,:);
Sigma0_em = theta_em.sigma(:,:,1);
Sigma1_em = theta_em.sigma(:,:,1);
w0_em = theta_em.w(1);
w1_em = theta_em.w(2);

pdf_0_em = @(X) w0_em*(mvnpdf(X, mu0_em, Sigma0_em));
pdf_1_em = @(X) w1_em*(mvnpdf(X, mu1_em, Sigma1_em));

eta_em = @(X) (pdf_0_em(X))./(pdf_0_em(X) + pdf_1_em(X));
EM_class = (eta_em(MIXED_sample) <= 1/2);
                    
% Errors - Type 1 and Type 2:
error_Typ1_EM = logical((EM_class ~= MIXED_class).*MIXED_class);
error_Typ2_EM = logical((EM_class ~= MIXED_class).*(1-MIXED_class));
EM_emp_risk = (sum(error_Typ1_EM) + sum(error_Typ2_EM))/n;

disp(' ')
disp(' EM empirical risk:')
disp(' ')
disp([' r = ', num2str(EM_emp_risk)])
disp(' ')

% Plot:
figure(fig_id)
fig_id = fig_id+1;
hold on
title('EM classifier error')
plot(ILL_subsample(:,1), ILL_subsample(:,2), 'k+')
plot(HEALTHY_subsample(:,1), HEALTHY_subsample(:,2), 'ko')
plot(MIXED_sample(error_Typ1_EM,1), MIXED_sample(error_Typ1_EM,2), 'rx', 'MarkerSize', 12)
plot(MIXED_sample(error_Typ2_EM,1), MIXED_sample(error_Typ2_EM,2), 'gx', 'MarkerSize', 12)
xlim(lim_x)
ylim(lim_y)
bound_image = getDecisionBoundaryPlot(lim_x, lim_y, ...
              @(x) eta_em(x)-1/2, 1);
colormap parula
imagesc(lim_x, lim_y, bound_image, 'AlphaData', 0.2)
grid on
legend('ILL sample', 'HEALTHY sample', 'Type 1 errors', 'Type 2 errors', ...
            'Location', 'best')
hold off
            

% PDF 3D plot:
x1 = linspace(lim_x(1), lim_x(2), 100);
x2 = linspace(lim_y(1), lim_y(2), 100);
[X1, X2] = meshgrid(x1, x2);
pdf_mixed = reshape(pdf_0_em([X1(:), X2(:)]) + pdf_1_em([X1(:), X2(:)]), 100, 100);
figure(fig_id)
fig_id = fig_id+1;

hold on
title('EM density estimation')
plot(ILL_subsample(:,1), ILL_subsample(:,2), 'b+')
plot(HEALTHY_subsample(:,1), HEALTHY_subsample(:,2), 'ro')            
surf(x1, x2, pdf_mixed, 'FaceAlpha', 0.5);
xlim(lim_x)
ylim(lim_y)
shading interp
colormap jet
pbaspect([1 1 0.5])
view(3)
grid on
legend('ILL sample', 'HEALTHY sample', 'MLE mixed density', ...
            'Location', 'best')

%% 12.- Generate a new testing sample and compute the testing errors of all the above
% classifiers

% TEST

n = 1000;

test_bern = binornd(1, p_mixture, n, 1);

mu1 = [mu_x1_1; mu_x2_1];
Sigma1 = [sigma2_x1_1, sigma_x1x2_1; sigma_x1x2_1, sigma2_x2_1];
mu0 = [mu_x1_0; mu_x2_0];
Sigma0 = [sigma2_x1_0, sigma_x1x2_0; sigma_x1x2_0, sigma2_x2_0];

ILL_TEST = mvnrnd(mu1, Sigma1, n);
ILL_TEST = ILL_TEST(test_bern == 1, :);
HEALTHY_TEST = mvnrnd(mu0, Sigma0, n);
HEALTHY_TEST = HEALTHY_TEST(test_bern == 0, :);

MIXED_TEST = [ILL_TEST; HEALTHY_TEST];
MIXED_TEST_class = [ones(size(ILL_TEST, 1), 1); ...
                        zeros(size(HEALTHY_TEST, 1), 1)];

n_component1 = size(ILL_TEST, 1);
n_component0 = size(HEALTHY_TEST, 1);
                    
% Figure limits:
xl_t = [min(MIXED_TEST(:,1))-10, max(MIXED_TEST(:,1))+10];
yl_t = [min(MIXED_TEST(:,2))-10, max(MIXED_TEST(:,2))+10];

% Bayes:
bayes_ill = MIXED_TEST(eta_f(MIXED_TEST) > 0, :);
bayes_healthy = MIXED_TEST(eta_f(MIXED_TEST) <= 0, :);

% Errors - Type 1 and Type 2:
error_Typ1_bayes = setdiff(bayes_ill, ILL_TEST, 'rows');
error_Typ2_bayes = setdiff(bayes_healthy, HEALTHY_TEST, 'rows');
bayes_emp_risk = (length(error_Typ1_bayes) + length(error_Typ2_bayes))/n;

% Logistic:
logit_ill = MIXED_TEST(logit_crit(MIXED_TEST, log_beta) > 1, :);
logit_healthy = MIXED_TEST(logit_crit(MIXED_TEST, log_beta) <= 1, :);
% Errors - Type 1 and Type 2:
error_Typ1_logit = setdiff(logit_ill, ILL_TEST, 'rows');
error_Typ2_logit = setdiff(logit_healthy, HEALTHY_TEST, 'rows');
logit_emp_risk = (length(error_Typ1_logit) + length(error_Typ2_logit))/n;

% LDA:
delta_ill = delta_r_lda(MIXED_TEST, w1_lda, mu1_lda, Sigma_lda);
delta_healthy = delta_r_lda(MIXED_TEST, w0_lda, mu0_lda, Sigma_lda);
LDA_ill = MIXED_TEST(delta_ill > delta_healthy, :);
LDA_healthy = MIXED_TEST(delta_ill <= delta_healthy, :);
% Errors - Type 1 and Type 2:
error_Typ1_LDA = setdiff(LDA_ill, ILL_TEST, 'rows');
error_Typ2_LDA = setdiff(LDA_healthy, HEALTHY_TEST, 'rows');
LDA_emp_risk = (length(error_Typ1_LDA) + length(error_Typ2_LDA))/n;

% QDA:
delta_ill = delta_r_qda(MIXED_TEST, w1_qda, mu1_qda, Sigma12_qda);
delta_healthy = delta_r_qda(MIXED_TEST, w0_qda, mu0_qda, Sigma02_qda);
QDA_ill = MIXED_TEST(delta_ill > delta_healthy, :);
QDA_healthy = MIXED_TEST(delta_ill <= delta_healthy, :);
% Errors - Type 1 and Type 2:
error_Typ1_QDA = setdiff(QDA_ill, ILL_TEST, 'rows');
error_Typ2_QDA = setdiff(QDA_healthy, HEALTHY_TEST, 'rows');
QDA_emp_risk = (length(error_Typ1_QDA) + length(error_Typ2_QDA))/n;

% Naive Bayes:
delta_ill = delta_r_nbc(MIXED_TEST);
delta_healthy = 1 - delta_ill;
NBC_ill = MIXED_TEST(delta_ill > delta_healthy, :);
NBC_healthy = MIXED_TEST(delta_ill <= delta_healthy, :);
% Errors - Type 1 and Type 2:
error_Typ1_NBC = setdiff(NBC_ill, ILL_TEST, 'rows');
error_Typ2_NBC = setdiff(NBC_healthy, HEALTHY_TEST, 'rows');
NBC_emp_risk = (length(error_Typ1_NBC) + length(error_Typ2_NBC))/n;

% Perceptron:
h_sample_test = [ones(size(MIXED_TEST,1),1), MIXED_TEST];
percept_ill = MIXED_TEST((h_sample_test*beta_percept(:) <= 0), :);
percept_healthy = MIXED_TEST((h_sample_test*beta_percept(:) > 0), :);
% Errors - Type 1 and Type 2:
error_Typ1_percept = setdiff(percept_ill, ILL_TEST, 'rows');
error_Typ2_percept = setdiff(percept_healthy, HEALTHY_TEST, 'rows');
percept_emp_risk = (length(error_Typ1_percept) + length(error_Typ2_percept))/n;

% KNN:
knn_1_class = predict(knn_1, MIXED_TEST);
% Errors - Type 1 and Type 2:
error_Typ1_knn_1 = logical((knn_1_class ~= MIXED_TEST_class).*MIXED_TEST_class);
error_Typ2_knn_1 = logical(((1-knn_1_class) ~= (1-MIXED_TEST_class)).*(1-MIXED_TEST_class));
knn_1_emp_risk = (sum(error_Typ1_knn_1) + sum(error_Typ2_knn_1))/n;

knn_5_class = predict(knn_5, MIXED_TEST);
% Errors - Type 1 and Type 2:
error_Typ1_knn_5 = logical((knn_5_class ~= MIXED_TEST_class).*MIXED_TEST_class);
error_Typ2_knn_5 = logical(((1-knn_5_class) ~= (1-MIXED_TEST_class)).*(1-MIXED_TEST_class));
knn_5_emp_risk = (sum(error_Typ1_knn_5) + sum(error_Typ2_knn_5))/n;

knn_10_class = predict(knn_10, MIXED_TEST);
% Errors - Type 1 and Type 2:
error_Typ1_knn_10 = logical((knn_10_class ~= MIXED_TEST_class).*MIXED_TEST_class);
error_Typ2_knn_10 = logical((knn_10_class ~= MIXED_TEST_class).*(1-MIXED_TEST_class));
knn_10_emp_risk = (sum(error_Typ1_knn_10) + sum(error_Typ2_knn_10))/n;

% MLE:
MLE_class = (eta_mle(MIXED_TEST) <= 1/2);               
% Errors - Type 1 and Type 2:
error_Typ1_MLE = logical((MLE_class ~= MIXED_TEST_class).*MIXED_TEST_class);
error_Typ2_MLE = logical((MLE_class ~= MIXED_TEST_class).*(1-MIXED_TEST_class));
MLE_emp_risk = (sum(error_Typ1_MLE) + sum(error_Typ2_MLE))/n;

% EM:
EM_class = (eta_em(MIXED_TEST) <= 1/2);
% Errors - Type 1 and Type 2:
error_Typ1_EM = logical((EM_class ~= MIXED_TEST_class).*MIXED_TEST_class);
error_Typ2_EM = logical((EM_class ~= MIXED_TEST_class).*(1-MIXED_TEST_class));
EM_emp_risk = (sum(error_Typ1_EM) + sum(error_Typ2_EM))/n;

% Subplots of test results:
figure(fig_id)
fig_id = fig_id+1;

subplot(2,2,1)
hold on
title('Test samples')
plot(ILL_TEST(:,1), ILL_TEST(:,2), 'b+')
plot(HEALTHY_TEST(:,1), HEALTHY_TEST(:,2), 'ro')
grid on
legend('ILL sample', 'HEALTHY sample')
hold off

subplot(2,2,2)
hold on
title('Bayes classifier error')
plot(ILL_TEST(:,1), ILL_TEST(:,2), 'k+')
plot(HEALTHY_TEST(:,1), HEALTHY_TEST(:,2), 'ko')
plot(error_Typ1_bayes(:,1), error_Typ1_bayes(:,2), 'rx', 'MarkerSize', 12)
plot(error_Typ2_bayes(:,1), error_Typ2_bayes(:,2), 'gx', 'MarkerSize', 12)
xlim(xl_t)
ylim(yl_t)
bound_image = getDecisionBoundaryPlot(xl_t, yl_t, eta_f, 1);
colormap parula
imagesc(xl_t, yl_t, bound_image, 'AlphaData', 0.2)
grid on
legend('ILL sample', 'HEALTHY sample', 'Type 1 errors', 'Type 2 errors', ...
            'Location', 'best')
hold off

subplot(2,2,3)
hold on
title('Logistic classifier error')
plot(ILL_TEST(:,1), ILL_TEST(:,2), 'k+')
plot(HEALTHY_TEST(:,1), HEALTHY_TEST(:,2), 'ko')
plot(error_Typ1_logit(:,1), error_Typ1_logit(:,2), 'rx', 'MarkerSize', 12)
plot(error_Typ2_logit(:,1), error_Typ2_logit(:,2), 'gx', 'MarkerSize', 12)
xlim(xl_t)
ylim(yl_t)
bound_image = getDecisionBoundaryPlot(xl_t, yl_t, ...
                                    @(x) logit_crit(x, log_beta)-1, 1);
colormap parula
imagesc(xl_t, yl_t, bound_image, 'AlphaData', 0.2)
grid on
legend('ILL sample', 'HEALTHY sample', 'Type 1 errors', 'Type 2 errors', ...
            'Location', 'best')
hold off

subplot(2,2,4)
hold on
title('LDA classifier error')
plot(ILL_TEST(:,1), ILL_TEST(:,2), 'k+')
plot(HEALTHY_TEST(:,1), HEALTHY_TEST(:,2), 'ko')
plot(error_Typ1_LDA(:,1), error_Typ1_LDA(:,2), 'rx', 'MarkerSize', 12)
plot(error_Typ2_LDA(:,1), error_Typ2_LDA(:,2), 'gx', 'MarkerSize', 12)
xlim(xl_t)
ylim(yl_t)
bound_image = getDecisionBoundaryPlot(xl_t, yl_t, ...
              @(x) delta_r_lda(x, w1_lda, mu1_lda, Sigma_lda) - ...
                   delta_r_lda(x, w0_lda, mu0_lda, Sigma_lda), 1);
colormap parula
imagesc(xl_t, yl_t, bound_image, 'AlphaData', 0.2)
grid on
legend('ILL sample', 'HEALTHY sample', 'Type 1 errors', 'Type 2 errors', ...
            'Location', 'best')
hold off

figure(fig_id)
fig_id = fig_id+1;

subplot(2,2,1)
hold on
title('Test samples')
plot(ILL_TEST(:,1), ILL_TEST(:,2), 'b+')
plot(HEALTHY_TEST(:,1), HEALTHY_TEST(:,2), 'ro')
grid on
legend('ILL sample', 'HEALTHY sample')
hold off

subplot(2,2,2)
hold on
title('QDA classifier error')
plot(ILL_TEST(:,1), ILL_TEST(:,2), 'k+')
plot(HEALTHY_TEST(:,1), HEALTHY_TEST(:,2), 'ko')
plot(error_Typ1_QDA(:,1), error_Typ1_QDA(:,2), 'rx', 'MarkerSize', 12)
plot(error_Typ2_QDA(:,1), error_Typ2_QDA(:,2), 'gx', 'MarkerSize', 12)
xlim(xl_t)
ylim(yl_t)
bound_image = getDecisionBoundaryPlot(xl_t, yl_t, ...
              @(x) delta_r_qda(x, w1_qda, mu1_qda, Sigma12_qda) - ...
                   delta_r_qda(x, w0_qda, mu0_qda, Sigma02_qda), 1);
colormap parula
imagesc(xl_t, yl_t, bound_image, 'AlphaData', 0.2)
grid on
legend('ILL sample', 'HEALTHY sample', 'Type 1 errors', 'Type 2 errors', ...
            'Location', 'best')
hold off

subplot(2,2,3)
hold on
title('Naive Bayes classifier error')
plot(ILL_TEST(:,1), ILL_TEST(:,2), 'k+')
plot(HEALTHY_TEST(:,1), HEALTHY_TEST(:,2), 'ko')
plot(error_Typ1_NBC(:,1), error_Typ1_NBC(:,2), 'rx', 'MarkerSize', 12)
plot(error_Typ2_NBC(:,1), error_Typ2_NBC(:,2), 'gx', 'MarkerSize', 12)
xlim(xl_t)
ylim(yl_t)
bound_image = getDecisionBoundaryPlot(xl_t, yl_t, ...
              @(x) delta_r_nbc(x)-1/2, 1);
colormap parula
imagesc(xl_t, yl_t, bound_image, 'AlphaData', 0.2)
grid on
legend('ILL sample', 'HEALTHY sample', 'Type 1 errors', 'Type 2 errors', ...
            'Location', 'best')
hold off

subplot(2,2,4)
hold on
title('Perceptron classifier error')
plot(ILL_TEST(:,1), ILL_TEST(:,2), 'k+')
plot(HEALTHY_TEST(:,1), HEALTHY_TEST(:,2), 'ko')
plot(error_Typ1_percept(:,1), error_Typ1_percept(:,2), 'rx', 'MarkerSize', 12)
plot(error_Typ2_percept(:,1), error_Typ2_percept(:,2), 'gx', 'MarkerSize', 12)
xlim(xl_t)
ylim(yl_t)
bound_image = getDecisionBoundaryPlot(xl_t, yl_t, ...
              @(x) [ones(size(x,1),1), x]*beta_percept(:) < 0, 1);
colormap parula
imagesc(xl_t, yl_t, bound_image, 'AlphaData', 0.2)
grid on
legend('ILL sample', 'HEALTHY sample', 'Type 1 errors', 'Type 2 errors', ...
            'Location', 'best')
hold off

figure(fig_id)
fig_id = fig_id+1;

subplot(2,2,1)
hold on
title('Test samples')
plot(ILL_TEST(:,1), ILL_TEST(:,2), 'b+')
plot(HEALTHY_TEST(:,1), HEALTHY_TEST(:,2), 'ro')
grid on
legend('ILL sample', 'HEALTHY sample')
hold off

subplot(2,2,2)
hold on
title('K-Nearest Neighbors classifier error - 1 Neighbor')
plot(ILL_TEST(:,1), ILL_TEST(:,2), 'k+')
plot(HEALTHY_TEST(:,1), HEALTHY_TEST(:,2), 'ko')
plot(MIXED_TEST(error_Typ1_knn_1,1), MIXED_TEST(error_Typ1_knn_1,2), 'rx', 'MarkerSize', 12)
plot(MIXED_TEST(error_Typ2_knn_1,1), MIXED_TEST(error_Typ2_knn_1,2), 'gx', 'MarkerSize', 12)
xlim(xl_t)
ylim(yl_t)
bound_image = getDecisionBoundaryPlot(xl_t, yl_t, ...
              @(x) predict(knn_1, x), 1);
colormap parula
imagesc(xl_t, yl_t, bound_image, 'AlphaData', 0.2)
grid on
legend('ILL sample', 'HEALTHY sample', 'Type 1 errors', 'Type 2 errors', ...
            'Location', 'best')
hold off

subplot(2,2,3)
hold on
title('K-Nearest Neighbors classifier error - 5 Neighbor')
plot(ILL_TEST(:,1), ILL_TEST(:,2), 'k+')
plot(HEALTHY_TEST(:,1), HEALTHY_TEST(:,2), 'ko')
plot(MIXED_TEST(error_Typ1_knn_5,1), MIXED_TEST(error_Typ1_knn_5,2), 'rx', 'MarkerSize', 12)
plot(MIXED_TEST(error_Typ2_knn_5,1), MIXED_TEST(error_Typ2_knn_5,2), 'gx', 'MarkerSize', 12)
xlim(xl_t)
ylim(yl_t)
bound_image = getDecisionBoundaryPlot(xl_t, yl_t, ...
              @(x) predict(knn_5, x), 1);
colormap parula
imagesc(xl_t, yl_t, bound_image, 'AlphaData', 0.2)
grid on
legend('ILL sample', 'HEALTHY sample', 'Type 1 errors', 'Type 2 errors', ...
            'Location', 'best')
hold off

subplot(2,2,4)
hold on
title('K-Nearest Neighbors classifier error - 10 Neighbor')
plot(ILL_TEST(:,1), ILL_TEST(:,2), 'k+')
plot(HEALTHY_TEST(:,1), HEALTHY_TEST(:,2), 'ko')
plot(MIXED_TEST(error_Typ1_knn_10,1), MIXED_TEST(error_Typ1_knn_10,2), 'rx', 'MarkerSize', 12)
plot(MIXED_TEST(error_Typ2_knn_10,1), MIXED_TEST(error_Typ2_knn_10,2), 'gx', 'MarkerSize', 12)
xlim(xl_t)
ylim(yl_t)
bound_image = getDecisionBoundaryPlot(xl_t, yl_t, ...
              @(x) predict(knn_10, x), 1);
colormap parula
imagesc(xl_t, yl_t, bound_image, 'AlphaData', 0.2)
grid on
legend('ILL sample', 'HEALTHY sample', 'Type 1 errors', 'Type 2 errors', ...
            'Location', 'best')
hold off

figure(fig_id)
fig_id = fig_id+1;

subplot(1,3,1)
hold on
title('Test samples')
plot(ILL_TEST(:,1), ILL_TEST(:,2), 'b+')
plot(HEALTHY_TEST(:,1), HEALTHY_TEST(:,2), 'ro')
grid on
legend('ILL sample', 'HEALTHY sample')
hold off

subplot(1,3,2)
hold on
title('MLE classifier error')
plot(ILL_TEST(:,1), ILL_TEST(:,2), 'k+')
plot(HEALTHY_TEST(:,1), HEALTHY_TEST(:,2), 'ko')
plot(MIXED_TEST(error_Typ1_MLE,1), MIXED_TEST(error_Typ1_MLE,2), 'rx', 'MarkerSize', 12)
plot(MIXED_TEST(error_Typ2_MLE,1), MIXED_TEST(error_Typ2_MLE,2), 'gx', 'MarkerSize', 12)
xlim(xl_t)
ylim(yl_t)
bound_image = getDecisionBoundaryPlot(xl_t, yl_t, ...
              @(x) eta_mle(x)-1/2, 1);
colormap parula
imagesc(xl_t, yl_t, bound_image, 'AlphaData', 0.2)
grid on
legend('ILL sample', 'HEALTHY sample', 'Type 1 errors', 'Type 2 errors', ...
            'Location', 'best')
hold off

subplot(1,3,3)
hold on
title('EM classifier error')
plot(ILL_TEST(:,1), ILL_TEST(:,2), 'k+')
plot(HEALTHY_TEST(:,1), HEALTHY_TEST(:,2), 'ko')
plot(MIXED_TEST(error_Typ1_EM,1), MIXED_TEST(error_Typ1_EM,2), 'rx', 'MarkerSize', 12)
plot(MIXED_TEST(error_Typ2_EM,1), MIXED_TEST(error_Typ2_EM,2), 'gx', 'MarkerSize', 12)
xlim(xl_t)
ylim(yl_t)
bound_image = getDecisionBoundaryPlot(xl_t, yl_t, ...
              @(x) eta_em(x)-1/2, 1);
colormap parula
imagesc(xl_t, yl_t, bound_image, 'AlphaData', 0.2)
grid on
legend('ILL sample', 'HEALTHY sample', 'Type 1 errors', 'Type 2 errors', ...
            'Location', 'best')
hold off


%% 13.- Plot decision boundaries for the above used classifiers
% For that adapt the code in
% http://www.peteryu.ca/tutorials/matlab/visualize_decision_boundaries 
% for your needs

