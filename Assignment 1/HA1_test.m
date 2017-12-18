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


