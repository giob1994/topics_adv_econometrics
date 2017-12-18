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

figj = 1;

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

bern = binornd(1, p_mixture, n, 1);

mu1 = [mu_x1_1; mu_x2_1];
Sigma1 = [sigma2_x1_1, sigma_x1x2_1; sigma_x1x2_1, sigma2_x2_1];
mu0 = [mu_x1_0; mu_x2_0];
Sigma0 = [sigma2_x1_0, sigma_x1x2_0; sigma_x1x2_0, sigma2_x2_0];

ill_sample = mvnrnd(mu1, Sigma1, n);
ill_sample = ill_sample(bern == 1, :);
healthy_sample = mvnrnd(mu0, Sigma0, n);
healthy_sample = healthy_sample(bern == 0, :);

mixed_sample = [ill_sample; healthy_sample];
sample_class = [ones(size(ill_sample, 1), 1); ...
                        zeros(size(healthy_sample, 1), 1)];

n_component1 = size(ill_sample, 1);
n_component0 = size(healthy_sample, 1);
                    
% Figure limits:
xl = [min(mixed_sample(:,1))-10, max(mixed_sample(:,1))+10];
yl = [min(mixed_sample(:,2))-10, max(mixed_sample(:,2))+10];

%% 2.- Create a 2D scatter plot that represents the hemoglobine level and health of the
% anemic and healthy subjects. Denote the anemic subjects with the symbol '+'
% and with 'o' the healthy ones

figure(figj)
figj = figj+1;

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

figure(figj)
figj = figj+1;

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

figure(figj)
figj = figj+1;

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

w1_lda = n_component1/n;
w0_lda = n_component0/n;

mu1_lda = mean(ill_sample);
mu0_lda = mean(healthy_sample);

Sigma12_s = cov(ill_sample - mu1_lda);
Sigma02_s = cov(healthy_sample - mu0_lda);
Sigma_lda = (1 / (n-2)) * (Sigma12_s + Sigma02_s);
                            
delta_r_lda = @(X, w, mu, Sigma) X*(Sigma\mu')-1/2*(mu*(Sigma\mu'))+log(w);

delta_ill = delta_r_lda(mixed_sample, w1_lda, mu1_lda, Sigma_lda);
delta_healthy = delta_r_lda(mixed_sample, w0_lda, mu0_lda, Sigma_lda);

LDA_ill = mixed_sample(delta_ill > delta_healthy, :);
LDA_healthy = mixed_sample(delta_ill <= delta_healthy, :);

% Type 1 errors:
Type1_er_LDA = setdiff(LDA_ill, ill_sample, 'rows');
% Type 2 errors:
Type2_er_LDA = setdiff(LDA_healthy, healthy_sample, 'rows');

% Empirical risk:
LDA_emp_risk = (length(Type1_er_LDA) + length(Type2_er_LDA))/n;

% Plot:
figure(figj)
figj = figj+1;

hold on
title('LDA classifier error')

plot(ill_sample(:,1), ill_sample(:,2), 'k+')
plot(healthy_sample(:,1), healthy_sample(:,2), 'ko')
plot(Type1_er_LDA(:,1), Type1_er_LDA(:,2), 'rx', 'MarkerSize', 12)
plot(Type2_er_LDA(:,1), Type2_er_LDA(:,2), 'gx', 'MarkerSize', 12)
xlim(xl)
ylim(yl)

bound_image = getDecisionBoundaryPlot(xl, yl, ...
              @(x) delta_r_lda(x, w1_lda, mu1_lda, Sigma_lda) - ...
                   delta_r_lda(x, w0_lda, mu0_lda, Sigma_lda), 1);
colormap spring
imagesc(xl, yl, bound_image, 'AlphaData', 0.2)

grid on
legend('ILL sample', 'HEALTHY sample', 'Type 1 errors', 'Type 2 errors', ...
            'Location', 'best')
% pbaspect([2 2 1])
% legend('MIXED sample')
hold off


%% 8.- Construct the QDA classifier based on the training sample and
% compute the associated errors and classifier's empirical risk.

w1_qda = n_component1/n;
w0_qda = n_component0/n;

mu1_qda = mean(ill_sample);
mu0_qda = mean(healthy_sample);

Sigma12_qda = cov(ill_sample - mu1_qda);
Sigma02_qda = cov(healthy_sample - mu0_qda);
                            
delta_r_qda = @(X, w, mu, Sigma) - 1/2*log(det(Sigma)) ...
                                 - 1/2*sum((X-mu).*(Sigma\(X-mu)')', 2) ...
                                 + log(w) ;

delta_ill = delta_r_qda(mixed_sample, w1_qda, mu1_qda, Sigma12_qda);
delta_healthy = delta_r_qda(mixed_sample, w0_qda, mu0_qda, Sigma02_qda);

QDA_ill = mixed_sample(delta_ill > delta_healthy, :);
QDA_healthy = mixed_sample(delta_ill <= delta_healthy, :);

% Type 1 errors:
Type1_er_QDA = setdiff(QDA_ill, ill_sample, 'rows');
% Type 2 errors:
Type2_er_QDA = setdiff(QDA_healthy, healthy_sample, 'rows');

% Empirical risk:
QDA_emp_risk = (length(Type1_er_QDA) + length(Type2_er_QDA))/n;

% Plot:
figure(figj)
figj = figj+1;

hold on
title('QDA classifier error')

plot(ill_sample(:,1), ill_sample(:,2), 'k+')
plot(healthy_sample(:,1), healthy_sample(:,2), 'ko')
plot(Type1_er_QDA(:,1), Type1_er_QDA(:,2), 'rx', 'MarkerSize', 12)
plot(Type2_er_QDA(:,1), Type2_er_QDA(:,2), 'gx', 'MarkerSize', 12)
xlim(xl)
ylim(yl)

bound_image = getDecisionBoundaryPlot(xl, yl, ...
              @(x) delta_r_qda(x, w1_qda, mu1_qda, Sigma12_qda) - ...
                   delta_r_qda(x, w0_qda, mu0_qda, Sigma02_qda), 1);
colormap spring
imagesc(xl, yl, bound_image, 'AlphaData', 0.2)

grid on
legend('ILL sample', 'HEALTHY sample', 'Type 1 errors', 'Type 2 errors', ...
            'Location', 'best')
% pbaspect([2 2 1])
% legend('MIXED sample')
hold off


%% 7.- Construct the Naive Bayes classifier based on the training sample and 
% compute the associated errors and classifier's empirical risk.

w1_nbc = n_component1/n;
w0_nbc = n_component0/n;

mu1_nbc = mean(ill_sample);
mu0_nbc = mean(healthy_sample);

Sigma12_nbc = diag(cov(ill_sample - mu1_nbc));
Sigma02_nbc = diag(cov(healthy_sample - mu0_nbc));

delta_r_nbc = @(X) ...
     w1_nbc*normpdf(X(:,1), mu1_nbc(1), Sigma12_nbc(1)).* ... 
            normpdf(X(:,2), mu1_nbc(2), Sigma12_nbc(2)) ...
 ./ (w1_nbc*normpdf(X(:,1), mu1_nbc(1), Sigma12_nbc(1)).* ...
            normpdf(X(:,2), mu1_nbc(2), Sigma12_nbc(2)) ... 
   + w0_nbc*normpdf(X(:,1), mu0_nbc(1), Sigma02_nbc(1)).* ...
            normpdf(X(:,2), mu0_nbc(2), Sigma02_nbc(2)));

delta_ill = delta_r_nbc(mixed_sample);
delta_healthy = 1 - delta_ill;

NBC_ill = mixed_sample(delta_ill > delta_healthy, :);
NBC_healthy = mixed_sample(delta_ill <= delta_healthy, :);

% Type 1 errors:
Type1_er_NBC = setdiff(NBC_ill, ill_sample, 'rows');
% Type 2 errors:
Type2_er_NBC = setdiff(NBC_healthy, healthy_sample, 'rows');

% Empirical risk:
NBC_emp_risk = (length(Type1_er_NBC) + length(Type2_er_NBC))/n;

% Plot:
figure(figj)
figj = figj+1;

hold on
title('Naive Bayes classifier error')

plot(ill_sample(:,1), ill_sample(:,2), 'k+')
plot(healthy_sample(:,1), healthy_sample(:,2), 'ko')
plot(Type1_er_NBC(:,1), Type1_er_NBC(:,2), 'rx', 'MarkerSize', 12)
plot(Type2_er_NBC(:,1), Type2_er_NBC(:,2), 'gx', 'MarkerSize', 12)
xlim(xl)
ylim(yl)

bound_image = getDecisionBoundaryPlot(xl, yl, ...
              @(x) delta_r_nbc(x)-1/2, 1);
colormap spring
imagesc(xl, yl, bound_image, 'AlphaData', 0.2)

grid on
legend('ILL sample', 'HEALTHY sample', 'Type 1 errors', 'Type 2 errors', ...
            'Location', 'best')
% pbaspect([2 2 1])
% legend('MIXED sample')
hold off

%% 9.- Construct perceptron hyperplane classifier based on the training sample and 
% compute the classifier's empirical risk.
% The perceptron is specified by a vector beta = [beta0, beta1, beta2] that 
% determines a separating hyperplane beta0 + beta1 * x1 + beta2 * x2
% Start by constructing a function that computes the empirical error of this
% classifier as a function of beta.

h_sample = [ones(size(mixed_sample,1),1), mixed_sample];

percept_emp_error = @(beta) sum((h_sample*beta(:) < 0) == sample_class)/n;

beta_percept = ga(percept_emp_error, 3);

percept_ill = mixed_sample((h_sample*beta_percept(:) <= 0), :);
percept_healthy = mixed_sample((h_sample*beta_percept(:) > 0), :);

% Type 1 errors:
Type1_er_percept = setdiff(percept_ill, ill_sample, 'rows');
% Type 2 errors:
Type2_er_percept = setdiff(percept_healthy, healthy_sample, 'rows');

% Empirical risk:
percept_emp_risk = (length(Type1_er_percept) + length(Type2_er_percept))/n;

% Plot:
figure(figj)
figj = figj+1;

hold on
title('Perceptron classifier error')

plot(ill_sample(:,1), ill_sample(:,2), 'k+')
plot(healthy_sample(:,1), healthy_sample(:,2), 'ko')
plot(Type1_er_percept(:,1), Type1_er_percept(:,2), 'rx', 'MarkerSize', 12)
plot(Type2_er_percept(:,1), Type2_er_percept(:,2), 'gx', 'MarkerSize', 12)
xlim(xl)
ylim(yl)

bound_image = getDecisionBoundaryPlot(xl, yl, ...
              @(x) [ones(size(x,1),1), x]*beta_percept(:) < 0, 1);
colormap spring
imagesc(xl, yl, bound_image, 'AlphaData', 0.2)

grid on
legend('ILL sample', 'HEALTHY sample', 'Type 1 errors', 'Type 2 errors', ...
            'Location', 'best')
% pbaspect([2 2 1])
% legend('MIXED sample')
hold off

%% 10.- Classify the data with a KNN classifier and compute the associated errors 
% and classifier's empirical risk with 1, 5, and 10 neighbors

knn_1 = fitcknn(mixed_sample, sample_class ,'NumNeighbors', 1);
knn_1_class = predict(knn_1, mixed_sample);

% KNN 1 - Type 1 errors:
Type1_er_knn_1 = logical((knn_1_class ~= sample_class).*sample_class);
% KNN 1 - Type 2 errors:
Type2_er_knn_1 = logical(((1-knn_1_class) ~= (1-sample_class)).*(1-sample_class));

% KNN 1 - Empirical risk:
knn_1_emp_risk = (sum(Type1_er_knn_1) + sum(Type2_er_knn_1))/n;

% Plot:
figure(figj)
figj = figj+1;

hold on
title('K-Nearest Neighbors classifier error - 1 Neighbor')

plot(ill_sample(:,1), ill_sample(:,2), 'k+')
plot(healthy_sample(:,1), healthy_sample(:,2), 'ko')
plot(mixed_sample(Type1_er_knn_1,1), mixed_sample(Type1_er_knn_1,2), 'rx', 'MarkerSize', 12)
plot(mixed_sample(Type2_er_knn_1,1), mixed_sample(Type2_er_knn_1,2), 'gx', 'MarkerSize', 12)
xlim(xl)
ylim(yl)

bound_image = getDecisionBoundaryPlot(xl, yl, ...
              @(x) predict(knn_1, x), 1);
colormap spring
imagesc(xl, yl, bound_image, 'AlphaData', 0.2)

grid on
legend('ILL sample', 'HEALTHY sample', 'Type 1 errors', 'Type 2 errors', ...
            'Location', 'best')
% pbaspect([2 2 1])
% legend('MIXED sample')
hold off


knn_5 = fitcknn(mixed_sample, sample_class ,'NumNeighbors', 5);
knn_5_class = predict(knn_5, mixed_sample);

% KNN 5 - Type 1 errors:
Type1_er_knn_5 = logical((knn_5_class ~= sample_class).*sample_class);
% KNN 5 - Type 2 errors:
Type2_er_knn_5 = logical(((1-knn_5_class) ~= (1-sample_class)).*(1-sample_class));

% KNN 5 - Empirical risk:
knn_5_emp_risk = (sum(Type1_er_knn_5) + sum(Type2_er_knn_5))/n;

% Plot:
figure(figj)
figj = figj+1;

hold on
title('K-Nearest Neighbors classifier error - 5 Neighbor')

plot(ill_sample(:,1), ill_sample(:,2), 'k+')
plot(healthy_sample(:,1), healthy_sample(:,2), 'ko')
plot(mixed_sample(Type1_er_knn_5,1), mixed_sample(Type1_er_knn_5,2), 'rx', 'MarkerSize', 12)
plot(mixed_sample(Type2_er_knn_5,1), mixed_sample(Type2_er_knn_5,2), 'gx', 'MarkerSize', 12)
xlim(xl)
ylim(yl)

bound_image = getDecisionBoundaryPlot(xl, yl, ...
              @(x) predict(knn_5, x), 1);
colormap spring
imagesc(xl, yl, bound_image, 'AlphaData', 0.2)

grid on
legend('ILL sample', 'HEALTHY sample', 'Type 1 errors', 'Type 2 errors', ...
            'Location', 'best')
% pbaspect([2 2 1])
% legend('MIXED sample')
hold off

knn_10 = fitcknn(mixed_sample, sample_class ,'NumNeighbors', 10);
knn_10_class = predict(knn_10, mixed_sample);

% KNN 10 - Type 1 errors:
Type1_er_knn_10 = logical((knn_10_class ~= sample_class).*sample_class);
% KNN 10 - Type 2 errors:
Type2_er_knn_10 = logical((knn_10_class ~= sample_class).*(1-sample_class));

% KNN 10 - Empirical risk:
knn_10_emp_risk = (sum(Type1_er_knn_10) + sum(Type2_er_knn_10))/n;

% Plot:
figure(figj)
figj = figj+1;

hold on
title('K-Nearest Neighbors classifier error - 10 Neighbor')

plot(ill_sample(:,1), ill_sample(:,2), 'k+')
plot(healthy_sample(:,1), healthy_sample(:,2), 'ko')
plot(mixed_sample(Type1_er_knn_10,1), mixed_sample(Type1_er_knn_10,2), 'rx', 'MarkerSize', 12)
plot(mixed_sample(Type2_er_knn_10,1), mixed_sample(Type2_er_knn_10,2), 'gx', 'MarkerSize', 12)
xlim(xl)
ylim(yl)

bound_image = getDecisionBoundaryPlot(xl, yl, ...
              @(x) predict(knn_10, x), 1);
colormap spring
imagesc(xl, yl, bound_image, 'AlphaData', 0.2)

grid on
legend('ILL sample', 'HEALTHY sample', 'Type 1 errors', 'Type 2 errors', ...
            'Location', 'best')
% pbaspect([2 2 1])
% legend('MIXED sample')
hold off

%% 11.- Classify the data using the unsupervised approach providing the number of classes 2
% (for that make use of the EM and MLE techniques implemented earlier; for the MLE 
% technique make an improvement and guarantee PSD constraint by construction using the Cholesky
% decomposition of covariance matrix), compute the associated errors and classifier's empirical risk.

% MLE estimation:

dims = 2;

loglikeF = @(theta) -getBiGaussianMixLogLikelihood(mixed_sample, theta, dims);
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
MLE_class = (eta_mle(mixed_sample) <= 1/2);
                    
% Type 1 errors:
Type1_er_MLE = logical((MLE_class ~= sample_class).*sample_class);
% Type 2 errors:
Type2_er_MLE = logical((MLE_class ~= sample_class).*(1-sample_class));

% Empirical risk:
MLE_emp_risk = (sum(Type1_er_MLE) + sum(Type2_er_MLE))/n;

% Plot:
figure(figj)
figj = figj+1;

hold on
title('MLE classifier error')

plot(ill_sample(:,1), ill_sample(:,2), 'k+')
plot(healthy_sample(:,1), healthy_sample(:,2), 'ko')
plot(mixed_sample(Type1_er_MLE,1), mixed_sample(Type1_er_MLE,2), 'rx', 'MarkerSize', 12)
plot(mixed_sample(Type2_er_MLE,1), mixed_sample(Type2_er_MLE,2), 'gx', 'MarkerSize', 12)
xlim(xl)
ylim(yl)

bound_image = getDecisionBoundaryPlot(xl, yl, ...
              @(x) eta_mle(x)-1/2, 1);
colormap spring
imagesc(xl, yl, bound_image, 'AlphaData', 0.2)

grid on
legend('ILL sample', 'HEALTHY sample', 'Type 1 errors', 'Type 2 errors', ...
            'Location', 'best')
% pbaspect([2 2 1])
% legend('MIXED sample')
hold off
            

% PDF 3D plot:
x1 = linspace(xl(1), xl(2), 100);
x2 = linspace(yl(1), yl(2), 100);
[X1, X2] = meshgrid(x1, x2);
pdf_mixed = reshape(pdf_0_mle([X1(:), X2(:)]) + pdf_1_mle([X1(:), X2(:)]), ...
                    100, 100);

figure(figj)
figj = figj+1;

hold on
title('MLE density estimation')

plot(ill_sample(:,1), ill_sample(:,2), 'b+')
plot(healthy_sample(:,1), healthy_sample(:,2), 'ro')
                
surf(x1, x2, pdf_mixed, 'FaceAlpha', 0.5);
xlim(xl)
ylim(yl)

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

theta_em = fitMultiMixedGaussianEM(mixed_sample, theta_0, precision);

mu0_em = theta_em.mu(1,:);
mu1_em = theta_em.mu(2,:);
Sigma0_em = theta_em.sigma(:,:,1);
Sigma1_em = theta_em.sigma(:,:,1);
w0_em = theta_em.w(1);
w1_em = theta_em.w(2);

pdf_0_em = @(X) w0_em*(mvnpdf(X, mu0_em, Sigma0_em));
pdf_1_em = @(X) w1_em*(mvnpdf(X, mu1_em, Sigma1_em));

eta_em = @(X) (pdf_0_em(X))./(pdf_0_em(X) + pdf_1_em(X));
EM_class = (eta_em(mixed_sample) <= 1/2);
                    
% Type 1 errors:
Type1_er_EM = logical((EM_class ~= sample_class).*sample_class);
% Type 2 errors:
Type2_er_EM = logical((EM_class ~= sample_class).*(1-sample_class));

% Empirical risk:
EM_emp_risk = (sum(Type1_er_EM) + sum(Type2_er_EM))/n;

% Plot:
figure(figj)
figj = figj+1;

hold on
title('EM classifier error')

plot(ill_sample(:,1), ill_sample(:,2), 'k+')
plot(healthy_sample(:,1), healthy_sample(:,2), 'ko')
plot(mixed_sample(Type1_er_EM,1), mixed_sample(Type1_er_EM,2), 'rx', 'MarkerSize', 12)
plot(mixed_sample(Type2_er_EM,1), mixed_sample(Type2_er_EM,2), 'gx', 'MarkerSize', 12)
xlim(xl)
ylim(yl)

bound_image = getDecisionBoundaryPlot(xl, yl, ...
              @(x) eta_em(x)-1/2, 1);
colormap spring
imagesc(xl, yl, bound_image, 'AlphaData', 0.2)

grid on
legend('ILL sample', 'HEALTHY sample', 'Type 1 errors', 'Type 2 errors', ...
            'Location', 'best')
% pbaspect([2 2 1])
% legend('MIXED sample')
hold off
            

% PDF 3D plot:
x1 = linspace(xl(1), xl(2), 100);
x2 = linspace(yl(1), yl(2), 100);
[X1, X2] = meshgrid(x1, x2);
pdf_mixed = reshape(pdf_0_em([X1(:), X2(:)]) + pdf_1_em([X1(:), X2(:)]), ...
                    100, 100);

figure(figj)
figj = figj+1;

hold on
title('EM density estimation')

plot(ill_sample(:,1), ill_sample(:,2), 'b+')
plot(healthy_sample(:,1), healthy_sample(:,2), 'ro')
                
surf(x1, x2, pdf_mixed, 'FaceAlpha', 0.5);
xlim(xl)
ylim(yl)

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

ill_test_sample = mvnrnd(mu1, Sigma1, n);
ill_test_sample = ill_test_sample(test_bern == 1, :);
healthy_test_sample = mvnrnd(mu0, Sigma0, n);
healthy_test_sample = healthy_test_sample(test_bern == 0, :);

test_sample = [ill_test_sample; healthy_test_sample];
test_sample_class = [ones(size(ill_test_sample, 1), 1); ...
                        zeros(size(healthy_test_sample, 1), 1)];

n_component1 = size(ill_test_sample, 1);
n_component0 = size(healthy_test_sample, 1);
                    
% Figure limits:
xl_t = [min(test_sample(:,1))-10, max(test_sample(:,1))+10];
yl_t = [min(test_sample(:,2))-10, max(test_sample(:,2))+10];

% Bayes:
bayes_ill = test_sample(eta_f(test_sample) > 0, :);
bayes_healthy = test_sample(eta_f(test_sample) <= 0, :);

% Type 1 errors:
Type1_er_bayes = setdiff(bayes_ill, ill_test_sample, 'rows');
% Type 2 errors:
Type2_er_bayes = setdiff(bayes_healthy, healthy_test_sample, 'rows');

% Empirical risk:
bayes_emp_risk = (length(Type1_er_bayes) + length(Type2_er_bayes))/n;

% Logistic:
logit_ill = test_sample(logit_crit(test_sample, log_beta) > 1, :);
logit_healthy = test_sample(logit_crit(test_sample, log_beta) <= 1, :);

% Type 1 errors:
Type1_er_logit = setdiff(logit_ill, ill_test_sample, 'rows');
% Type 2 errors:
Type2_er_logit = setdiff(logit_healthy, healthy_test_sample, 'rows');

% Empirical risk:
logit_emp_risk = (length(Type1_er_logit) + length(Type2_er_logit))/n;

% LDA:

delta_ill = delta_r_lda(test_sample, w1_lda, mu1_lda, Sigma_lda);
delta_healthy = delta_r_lda(test_sample, w0_lda, mu0_lda, Sigma_lda);

LDA_ill = test_sample(delta_ill > delta_healthy, :);
LDA_healthy = test_sample(delta_ill <= delta_healthy, :);

% Type 1 errors:
Type1_er_LDA = setdiff(LDA_ill, ill_test_sample, 'rows');
% Type 2 errors:
Type2_er_LDA = setdiff(LDA_healthy, healthy_test_sample, 'rows');

% Empirical risk:
LDA_emp_risk = (length(Type1_er_LDA) + length(Type2_er_LDA))/n;

% QDA:

delta_ill = delta_r_qda(test_sample, w1_qda, mu1_qda, Sigma12_qda);
delta_healthy = delta_r_qda(test_sample, w0_qda, mu0_qda, Sigma02_qda);

QDA_ill = test_sample(delta_ill > delta_healthy, :);
QDA_healthy = test_sample(delta_ill <= delta_healthy, :);

% Type 1 errors:
Type1_er_QDA = setdiff(QDA_ill, ill_test_sample, 'rows');
% Type 2 errors:
Type2_er_QDA = setdiff(QDA_healthy, healthy_test_sample, 'rows');

% Empirical risk:
QDA_emp_risk = (length(Type1_er_QDA) + length(Type2_er_QDA))/n;

% Naive Bayes:

delta_ill = delta_r_nbc(test_sample);
delta_healthy = 1 - delta_ill;

NBC_ill = test_sample(delta_ill > delta_healthy, :);
NBC_healthy = test_sample(delta_ill <= delta_healthy, :);

% Type 1 errors:
Type1_er_NBC = setdiff(NBC_ill, ill_test_sample, 'rows');
% Type 2 errors:
Type2_er_NBC = setdiff(NBC_healthy, healthy_test_sample, 'rows');

% Empirical risk:
NBC_emp_risk = (length(Type1_er_NBC) + length(Type2_er_NBC))/n;

% Perceptron:

h_sample_test = [ones(size(test_sample,1),1), test_sample];

percept_ill = test_sample((h_sample_test*beta_percept(:) <= 0), :);
percept_healthy = test_sample((h_sample_test*beta_percept(:) > 0), :);

% Type 1 errors:
Type1_er_percept = setdiff(percept_ill, ill_test_sample, 'rows');
% Type 2 errors:
Type2_er_percept = setdiff(percept_healthy, healthy_test_sample, 'rows');

% Empirical risk:
percept_emp_risk = (length(Type1_er_percept) + length(Type2_er_percept))/n;

% KNN:

knn_1_class = predict(knn_1, test_sample);

% KNN 1 - Type 1 errors:
Type1_er_knn_1 = logical((knn_1_class ~= test_sample_class).*test_sample_class);
% KNN 1 - Type 2 errors:
Type2_er_knn_1 = logical(((1-knn_1_class) ~= (1-test_sample_class)).*(1-test_sample_class));

% KNN 1 - Empirical risk:
knn_1_emp_risk = (sum(Type1_er_knn_1) + sum(Type2_er_knn_1))/n;

knn_5_class = predict(knn_5, test_sample);

% KNN 5 - Type 1 errors:
Type1_er_knn_5 = logical((knn_5_class ~= test_sample_class).*test_sample_class);
% KNN 5 - Type 2 errors:
Type2_er_knn_5 = logical(((1-knn_5_class) ~= (1-test_sample_class)).*(1-test_sample_class));

% KNN 5 - Empirical risk:
knn_5_emp_risk = (sum(Type1_er_knn_5) + sum(Type2_er_knn_5))/n;

knn_10_class = predict(knn_10, test_sample);

% KNN 10 - Type 1 errors:
Type1_er_knn_10 = logical((knn_10_class ~= test_sample_class).*test_sample_class);
% KNN 10 - Type 2 errors:
Type2_er_knn_10 = logical((knn_10_class ~= test_sample_class).*(1-test_sample_class));

% KNN 10 - Empirical risk:
knn_10_emp_risk = (sum(Type1_er_knn_10) + sum(Type2_er_knn_10))/n;

% MLE:
MLE_class = (eta_mle(test_sample) <= 1/2);
                    
% Type 1 errors:
Type1_er_MLE = logical((MLE_class ~= test_sample_class).*test_sample_class);
% Type 2 errors:
Type2_er_MLE = logical((MLE_class ~= test_sample_class).*(1-test_sample_class));

% Empirical risk:
MLE_emp_risk = (sum(Type1_er_MLE) + sum(Type2_er_MLE))/n;

% EM:
EM_class = (eta_em(test_sample) <= 1/2);
                    
% Type 1 errors:
Type1_er_EM = logical((EM_class ~= test_sample_class).*test_sample_class);
% Type 2 errors:
Type2_er_EM = logical((EM_class ~= test_sample_class).*(1-test_sample_class));

% Empirical risk:
EM_emp_risk = (sum(Type1_er_EM) + sum(Type2_er_EM))/n;

% Subplot of test results:
figure(figj)
figj = figj+1;

subplot(3,4,1)
hold on
title('Test samples')
plot(ill_test_sample(:,1), ill_test_sample(:,2), 'b+')
plot(healthy_test_sample(:,1), healthy_test_sample(:,2), 'ro')
grid on
% pbaspect([2 2 1])
legend('ILL sample', 'HEALTHY sample')
hold off

subplot(3,4,2)
hold on
title('Bayes classifier error')
plot(ill_test_sample(:,1), ill_test_sample(:,2), 'k+')
plot(healthy_test_sample(:,1), healthy_test_sample(:,2), 'ko')
plot(Type1_er_bayes(:,1), Type1_er_bayes(:,2), 'rx', 'MarkerSize', 12)
plot(Type2_er_bayes(:,1), Type2_er_bayes(:,2), 'gx', 'MarkerSize', 12)
xlim(xl_t)
ylim(yl_t)

bound_image = getDecisionBoundaryPlot(xl_t, yl_t, eta_f, 1);
colormap spring
imagesc(xl_t, yl_t, bound_image, 'AlphaData', 0.2)

grid on
legend('ILL sample', 'HEALTHY sample', 'Type 1 errors', 'Type 2 errors', ...
            'Location', 'best')
% pbaspect([2 2 1])
% legend('MIXED sample')
hold off

subplot(3,4,3)
hold on
title('Logistic classifier error')

plot(ill_test_sample(:,1), ill_test_sample(:,2), 'k+')
plot(healthy_test_sample(:,1), healthy_test_sample(:,2), 'ko')
plot(Type1_er_logit(:,1), Type1_er_logit(:,2), 'rx', 'MarkerSize', 12)
plot(Type2_er_logit(:,1), Type2_er_logit(:,2), 'gx', 'MarkerSize', 12)
xlim(xl_t)
ylim(yl_t)

bound_image = getDecisionBoundaryPlot(xl_t, yl_t, ...
                                    @(x) logit_crit(x, log_beta)-1, 1);
colormap spring
imagesc(xl_t, yl_t, bound_image, 'AlphaData', 0.2)

grid on
legend('ILL sample', 'HEALTHY sample', 'Type 1 errors', 'Type 2 errors', ...
            'Location', 'best')
% pbaspect([2 2 1])
% legend('MIXED sample')
hold off

subplot(3,4,4)
hold on
title('LDA classifier error')

plot(ill_test_sample(:,1), ill_test_sample(:,2), 'k+')
plot(healthy_test_sample(:,1), healthy_test_sample(:,2), 'ko')
plot(Type1_er_LDA(:,1), Type1_er_LDA(:,2), 'rx', 'MarkerSize', 12)
plot(Type2_er_LDA(:,1), Type2_er_LDA(:,2), 'gx', 'MarkerSize', 12)
xlim(xl_t)
ylim(yl_t)

bound_image = getDecisionBoundaryPlot(xl_t, yl_t, ...
              @(x) delta_r_lda(x, w1_lda, mu1_lda, Sigma_lda) - ...
                   delta_r_lda(x, w0_lda, mu0_lda, Sigma_lda), 1);
colormap spring
imagesc(xl_t, yl_t, bound_image, 'AlphaData', 0.2)

grid on
legend('ILL sample', 'HEALTHY sample', 'Type 1 errors', 'Type 2 errors', ...
            'Location', 'best')
% pbaspect([2 2 1])
% legend('MIXED sample')
hold off

subplot(3,4,5)
hold on
title('QDA classifier error')

plot(ill_test_sample(:,1), ill_test_sample(:,2), 'k+')
plot(healthy_test_sample(:,1), healthy_test_sample(:,2), 'ko')
plot(Type1_er_QDA(:,1), Type1_er_QDA(:,2), 'rx', 'MarkerSize', 12)
plot(Type2_er_QDA(:,1), Type2_er_QDA(:,2), 'gx', 'MarkerSize', 12)
xlim(xl_t)
ylim(yl_t)

bound_image = getDecisionBoundaryPlot(xl_t, yl_t, ...
              @(x) delta_r_qda(x, w1_qda, mu1_qda, Sigma12_qda) - ...
                   delta_r_qda(x, w0_qda, mu0_qda, Sigma02_qda), 1);
colormap spring
imagesc(xl_t, yl_t, bound_image, 'AlphaData', 0.2)

grid on
legend('ILL sample', 'HEALTHY sample', 'Type 1 errors', 'Type 2 errors', ...
            'Location', 'best')
% pbaspect([2 2 1])
% legend('MIXED sample')
hold off

subplot(3,4,6)
hold on
title('Naive Bayes classifier error')

plot(ill_test_sample(:,1), ill_test_sample(:,2), 'k+')
plot(healthy_test_sample(:,1), healthy_test_sample(:,2), 'ko')
plot(Type1_er_NBC(:,1), Type1_er_NBC(:,2), 'rx', 'MarkerSize', 12)
plot(Type2_er_NBC(:,1), Type2_er_NBC(:,2), 'gx', 'MarkerSize', 12)
xlim(xl_t)
ylim(yl_t)

bound_image = getDecisionBoundaryPlot(xl_t, yl_t, ...
              @(x) delta_r_nbc(x)-1/2, 1);
colormap spring
imagesc(xl_t, yl_t, bound_image, 'AlphaData', 0.2)

grid on
legend('ILL sample', 'HEALTHY sample', 'Type 1 errors', 'Type 2 errors', ...
            'Location', 'best')
% pbaspect([2 2 1])
% legend('MIXED sample')
hold off

subplot(3,4,7)
hold on
title('Perceptron classifier error')

plot(ill_test_sample(:,1), ill_test_sample(:,2), 'k+')
plot(healthy_test_sample(:,1), healthy_test_sample(:,2), 'ko')
plot(Type1_er_percept(:,1), Type1_er_percept(:,2), 'rx', 'MarkerSize', 12)
plot(Type2_er_percept(:,1), Type2_er_percept(:,2), 'gx', 'MarkerSize', 12)
xlim(xl_t)
ylim(yl_t)

bound_image = getDecisionBoundaryPlot(xl_t, yl_t, ...
              @(x) [ones(size(x,1),1), x]*beta_percept(:) < 0, 1);
colormap spring
imagesc(xl_t, yl_t, bound_image, 'AlphaData', 0.2)

grid on
legend('ILL sample', 'HEALTHY sample', 'Type 1 errors', 'Type 2 errors', ...
            'Location', 'best')
% pbaspect([2 2 1])
% legend('MIXED sample')
hold off

subplot(3,4,8)
hold on
title('K-Nearest Neighbors classifier error - 1 Neighbor')

plot(ill_test_sample(:,1), ill_test_sample(:,2), 'k+')
plot(healthy_test_sample(:,1), healthy_test_sample(:,2), 'ko')
plot(test_sample(Type1_er_knn_1,1), test_sample(Type1_er_knn_1,2), 'rx', 'MarkerSize', 12)
plot(test_sample(Type2_er_knn_1,1), test_sample(Type2_er_knn_1,2), 'gx', 'MarkerSize', 12)
xlim(xl_t)
ylim(yl_t)

bound_image = getDecisionBoundaryPlot(xl_t, yl_t, ...
              @(x) predict(knn_1, x), 1);
colormap spring
imagesc(xl_t, yl_t, bound_image, 'AlphaData', 0.2)

grid on
legend('ILL sample', 'HEALTHY sample', 'Type 1 errors', 'Type 2 errors', ...
            'Location', 'best')
% pbaspect([2 2 1])
% legend('MIXED sample')
hold off

subplot(3,4,9)
hold on
title('K-Nearest Neighbors classifier error - 5 Neighbor')

plot(ill_test_sample(:,1), ill_test_sample(:,2), 'k+')
plot(healthy_test_sample(:,1), healthy_test_sample(:,2), 'ko')
plot(test_sample(Type1_er_knn_5,1), test_sample(Type1_er_knn_5,2), 'rx', 'MarkerSize', 12)
plot(test_sample(Type2_er_knn_5,1), test_sample(Type2_er_knn_5,2), 'gx', 'MarkerSize', 12)
xlim(xl_t)
ylim(yl_t)

bound_image = getDecisionBoundaryPlot(xl_t, yl_t, ...
              @(x) predict(knn_5, x), 1);
colormap spring
imagesc(xl_t, yl_t, bound_image, 'AlphaData', 0.2)

grid on
legend('ILL sample', 'HEALTHY sample', 'Type 1 errors', 'Type 2 errors', ...
            'Location', 'best')
% pbaspect([2 2 1])
% legend('MIXED sample')
hold off

subplot(3,4,10)
hold on
title('K-Nearest Neighbors classifier error - 10 Neighbor')

plot(ill_test_sample(:,1), ill_test_sample(:,2), 'k+')
plot(healthy_test_sample(:,1), healthy_test_sample(:,2), 'ko')
plot(test_sample(Type1_er_knn_10,1), test_sample(Type1_er_knn_10,2), 'rx', 'MarkerSize', 12)
plot(test_sample(Type2_er_knn_10,1), test_sample(Type2_er_knn_10,2), 'gx', 'MarkerSize', 12)
xlim(xl_t)
ylim(yl_t)

bound_image = getDecisionBoundaryPlot(xl_t, yl_t, ...
              @(x) predict(knn_10, x), 1);
colormap spring
imagesc(xl_t, yl_t, bound_image, 'AlphaData', 0.2)

grid on
legend('ILL sample', 'HEALTHY sample', 'Type 1 errors', 'Type 2 errors', ...
            'Location', 'best')
% pbaspect([2 2 1])
% legend('MIXED sample')
hold off

subplot(3,4,11)
hold on
title('MLE classifier error')

plot(ill_test_sample(:,1), ill_test_sample(:,2), 'k+')
plot(healthy_test_sample(:,1), healthy_test_sample(:,2), 'ko')
plot(test_sample(Type1_er_MLE,1), test_sample(Type1_er_MLE,2), 'rx', 'MarkerSize', 12)
plot(test_sample(Type2_er_MLE,1), test_sample(Type2_er_MLE,2), 'gx', 'MarkerSize', 12)
xlim(xl_t)
ylim(yl_t)

bound_image = getDecisionBoundaryPlot(xl_t, yl_t, ...
              @(x) eta_mle(x)-1/2, 1);
colormap spring
imagesc(xl_t, yl_t, bound_image, 'AlphaData', 0.2)

grid on
legend('ILL sample', 'HEALTHY sample', 'Type 1 errors', 'Type 2 errors', ...
            'Location', 'best')
% pbaspect([2 2 1])
% legend('MIXED sample')
hold off

subplot(3,4,12)
hold on
title('EM classifier error')

plot(ill_test_sample(:,1), ill_test_sample(:,2), 'k+')
plot(healthy_test_sample(:,1), healthy_test_sample(:,2), 'ko')
plot(test_sample(Type1_er_EM,1), test_sample(Type1_er_EM,2), 'rx', 'MarkerSize', 12)
plot(test_sample(Type2_er_EM,1), test_sample(Type2_er_EM,2), 'gx', 'MarkerSize', 12)
xlim(xl_t)
ylim(yl_t)

bound_image = getDecisionBoundaryPlot(xl_t, yl_t, ...
              @(x) eta_em(x)-1/2, 1);
colormap spring
imagesc(xl_t, yl_t, bound_image, 'AlphaData', 0.2)

grid on
legend('ILL sample', 'HEALTHY sample', 'Type 1 errors', 'Type 2 errors', ...
            'Location', 'best')
% pbaspect([2 2 1])
% legend('MIXED sample')
hold off


%% 13.- Plot decision boundaries for the above used classifiers
% For that adapt the code in
% http://www.peteryu.ca/tutorials/matlab/visualize_decision_boundaries 
% for your needs

