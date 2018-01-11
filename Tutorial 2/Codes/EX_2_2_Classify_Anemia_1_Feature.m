%% ANEMIA CLASSIFICATION - 1 FEATURE CASE
% 
% The objective of this exercise is constructing several classifiers for
% anemia using the blood hemoglobin concentrations of the subjects under
% consideration. 
% X denotes the concentration of hemoglobin in grams/liter
% Y=1 means 'anemic', Y=0 means 'healthy'
% There are hence only two classes
% Goal: classify data into two classes based on one feature
%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear 
close all
rng default;

%The dataset is simulated out of a Gaussian mixture as follows
% Let {X|Y=1}~N(mu1, sigma12)
% Let {X|Y=0}~N(mu0, sigma02)

% Variables:
% n sample size
% mu1, mu0 means of the Gaussians used in the mixture
% sigma12, sigma02 variances of the Gaussians used in the mixture
% w1, w0 weights for each component of the mixture

n = 1000;
p_mixture = 1/4;

w1 = p_mixture;
w2 = 1 - p_mixture;

mu1 = 100;
sigma12 = 700;

mu0 = 155;
sigma02 = 500;


%% 1.-Simulate a training data set (x_i, y_i) with n elements 
% using Matlab binornd function

bern = binornd(1, p_mixture, n, 1);

ill_sample_1 = mu1 + sqrt(sigma12) * randn(n,1);
ill_sample_1 = ill_sample_1(bern == 1);
healthy_sample_0 = mu0 + sqrt(sigma02) * randn(n,1);
healthy_sample_0 = healthy_sample_0(bern == 0);

mix_sample = [ill_sample_1; healthy_sample_0];
sample_division = [ones(size(ill_sample_1)); ...
                        zeros(size(healthy_sample_0))];
% sample(bern == 1) = ill_sample_1(bern == 1);
% sample(bern == 0) = healthy_sample_0(bern == 0);

n_component1 = sum(bern);
n_component0 = n - n_component1;

% mu = [mu1; mu0];
% sigma = cat(3, (sigma12), (sigma22));
% p = [w1; w2];
% obj = gmdistribution(mu, sigma, p);
% Gauss_mix_sample = random(obj, n);


%% 2.- In the same figure plot:
% (1)-A count density-normalized histogram of the total subjects sample
% (The height of each bar is number of observations in bin / width of bin. 
% The area (height * width) of each bar is the number of observations in the bin. 
% The sum of the bar areas is the sample size)  
% (2)-A count density-normalized histogram of the healthy subjects sample 
% (3)-A count density-normalized histogram of the sick subjects sample
% (4)-Corresponding weighted versions of the two Gaussians used in the generation
% (5)-Corresponding weighted version of the Gaussian mixture distribution

mean_mixture = w1*mu1 + w2*mu0;
sigma2_mixture = w1*((mu1 - mean_mixture)^2+sigma12) + ...
                    w2*((mu0 - mean_mixture)^2+sigma02);

figure(1)
hold on

granularity = 100;

h = histogram(mix_sample, 'BinWidth', 10, 'Normalization', 'countdensity');
alpha(h, 0.6)

h1 = histogram(ill_sample_1, 'BinWidth', 10, 'Normalization', 'countdensity');
alpha(h1, 0.4)
h2 = histogram(healthy_sample_0, 'BinWidth', 10,  'Normalization', 'countdensity');
alpha(h2, 0.4)

% Plot both the Gaussian pdf with realtive weight:
pdf_1 = makedist('Normal', mu1, sqrt(sigma12));
pdf_2 = makedist('Normal', mu0, sqrt(sigma02));

% normPDF = @(x, m, s) 1/sqrt(2*pi*s))*exp(-(x-m).^2/(2*s);

x = linspace(mean_mixture - 3*sqrt(sigma2_mixture), ...
                mean_mixture + 3*sqrt(sigma2_mixture), granularity);

y1 = n_component1*pdf(pdf_1, x);
y2 = n_component0*pdf(pdf_2, x);

y_mixed = (y1 + y2);

cmap_w = winter(3);
cmap_h = autumn(3);

plot(x, y1, 'LineWidth', 2, 'Color', cmap_h(2,:))
plot(x, y2, 'LineWidth', 2, 'Color', cmap_w(2,:))
plot(x, y_mixed, 'LineWidth', 2, 'Color', 'k', 'LineStyle', ':')

legend('MIXED', 'Sample = ILL', 'Sample = HEALTHY', ...
       'pdf - ILL', 'pdf - HEALTHY', 'pdf - MIXED', 'Location', 'best')
   
grid on

hold off


%% 3. - Bayes classifier regression function and risk:
% (1) - Assuming that the data generating process is known, construct and  
% represent graphically the function eta that determines the Bayes classifier
% (2) - Compute the Bayes' risk according to formulas from Lecture 2

eta = @(x) (w1.*pdf(pdf_1, x))./(w1.*pdf(pdf_1, x) + w2.*pdf(pdf_2, x));

threshold = fzero(@(x) eta(x)-1/2, min(mu1, mu0)+abs(mu1-mu0)/2);

if mu0 > mu1
    
    bayes_risk = w2*cdf(pdf_2, threshold) + w1*(1-cdf(pdf_1, threshold));

else
    
    bayes_risk = w1*cdf(pdf_1, threshold) + w2*(1-cdf(pdf_2, threshold));
    
end

disp(' ')
disp(' Theoretical results:')
disp(' ')
disp([' Bayes risk:   ', num2str(bayes_risk)]);

       
x = linspace(-50, 300, 1000);

figure(2)
hold on

plot(x, eta(x), 'LineWidth', 2)
x_l = xlim;
y_l = ylim;
line(x_l, [1/2, 1/2], 'Color', 'r', 'LineStyle', ':', 'LineWidth', 1.2)
line([threshold, threshold], y_l, 'Color', 'k')
grid on
legend('\eta(x)', '1/2 decision boundary', ['Decision threshold = ',...
            num2str(threshold)], 'Location', 'best')
title('Bayes regression function \eta(x)')

hold off


%% 4.- Bayes critical value:
% Determine the critical value of the Bayes classifier by determining the
% hemoglobine concentration x_critical that yields eta(x_critical). Depict this value
% in the first figure

figure(1)
l_y = ylim;
line([threshold, threshold], l_y, 'Color', 'r')


%% 5.- Compute the number of Type I and Type II errors and the
% empirical risk of the Bayes classifier

% Type I errors:

if mu0 > mu1
    typ1 = sum(healthy_sample_0 <= threshold);
else 
    typ1 = sum(ill_sample_1 <= threshold);
end

% Type II errors:

if mu0 > mu1
    typ2 = sum(ill_sample_1 >= threshold);
else 
    typ2 = sum(healthy_sample_0 >= threshold);
end

emp_bayes_risk = (typ1 + typ2)/n;

disp([' Empirical Bayes risk:   ', num2str(emp_bayes_risk)])
disp(' ')


%% 6.-Demonstrate that empirical risk of Bayes' classifier converges to the
% true Bayes' risk provided that n->\infty

k = 20;

stride = 500;

emp_bayes_risk_ = zeros(1, length(1000:stride:10000));

t = 1;

for N = 1000:stride:100000

    tmp_emp_b = zeros(1, k);
    
    for i = 1:k
    
        bern_ = binornd(1, p_mixture, N, 1);

        ill_sample_1_ = mu1 + sqrt(sigma12) * randn(N,1);
        ill_sample_1_ = ill_sample_1_(bern_ == 1);
        healthy_sample_0_ = mu0 + sqrt(sigma02) * randn(N,1);
        healthy_sample_0_ = healthy_sample_0_(bern_ == 0);

        if mu0 > mu1
            typ1_ = sum(healthy_sample_0_ <= threshold);
            typ2_ = sum(ill_sample_1_ >= threshold);
        else 
            typ1_ = sum(ill_sample_1_ <= threshold);
            typ2_ = sum(healthy_sample_0_ >= threshold);
        end

        tmp_emp_b(i) = (typ1_ + typ2_)/N;
    
%     figure
%     hold on
% 
%     granularity = 100;
% 
%     h1 = histogram(ill_sample_1_, 'Normalization', 'countdensity');
%     alpha(h1, 0.7)
%     h2 = histogram(healthy_sample_0_,  'Normalization', 'countdensity');
%     alpha(h2, 0.7)
    
    
    end
    
    emp_bayes_risk_(t) = sum(tmp_emp_b)/k;
    
    t = t+1;
    
end

% Only for N->+INF

emp_bayes_risk_N = zeros(1, length(1000:stride:10000));

t = 1;

for N = 1000:stride:100000
    
    bern_ = binornd(1, p_mixture, N, 1);

    ill_sample_1_ = mu1 + sqrt(sigma12) * randn(N,1);
    ill_sample_1_ = ill_sample_1_(bern_ == 1);
    healthy_sample_0_ = mu0 + sqrt(sigma02) * randn(N,1);
    healthy_sample_0_ = healthy_sample_0_(bern_ == 0);

    if mu0 > mu1
        typ1_ = sum(healthy_sample_0_ <= threshold);
        typ2_ = sum(ill_sample_1_ >= threshold);
    else 
        typ1_ = sum(ill_sample_1_ <= threshold);
        typ2_ = sum(healthy_sample_0_ >= threshold);
    end

    emp_bayes_risk_N(t) = (typ1_ + typ2_)/N;
    
    t = t+1;
    
end

figure(3)

hold on
plot((1000:stride:100000), emp_bayes_risk_, '-o')
plot((1000:stride:100000), emp_bayes_risk_N, '-x')
x_l = xlim;
line(x_l, [bayes_risk, bayes_risk], 'Color', 'r', 'LineStyle', '--')
grid on
legend({'$\widehat{R}_{n}(f)$ - consistent simulation', ...
        '$\widehat{R}_{n}(f)$ - $n\rightarrow+\infty$', '$R_{n}(f)$'}, ...
            'Interpreter', 'latex', 'FontSize', 12)
title('Empirical Bayes risk plot')

hold off


% Practical classifiers

%% 7.- Construct the LDA classifier associated to the previous sample and 
% compute the associated errors and classifier risk

eta = @(x, pdf1, pdf2, w1, w2) (w1.*pdf(pdf1, x))./ ... 
                                (w1.*pdf(pdf1, x) + w2.*pdf(pdf2, x));

thresh_f = @(x, pdf1, pdf2, m1, m0, w1, w2) ...
            fzero(@(x) (eta(x, pdf1, pdf2, w1, w2) - 1/2), ...
                        min(m1,m0)+abs(m1-m0)/2);

% The LDA classifier requires to estimate the Gaussian parameters: we use
% MLE estimation:
%
% NOTE: LDA has as hypothesis that sigma12 == sigma02, thus we will need
%       to use 'fmincon'. Also, these are NOT the same estimates that
%       will be obtained by using GMModel in point (9) for the 
%       unsupervised MLE estimation.

w1_lda = n_component1/n;
w0_lda = n_component0/n;

mu1_lda = mean(ill_sample_1);
mu0_lda = mean(healthy_sample_0);

sigma12_s = sum((ill_sample_1 - mu1_lda).^2);
sigma02_s = sum((healthy_sample_0 - mu0_lda).^2);
sigma_lda = (1 / (n-2)) * (sigma12_s + sigma02_s);

disp(' ')
disp(' MLE estimation of Gaussian mixture:')
disp(' ')
disp([' mu1 = ', num2str(mu1_lda), '   mu0 = ', num2str(mu0_lda)])
disp([' sigma12 = ', num2str(sigma_lda), ...
                '   sigma02 = ', num2str(sigma_lda)])
disp([' w1 = ', num2str(w1_lda), '   w0 = ', num2str(w0_lda)])
disp(' ')


figure(4)
hold on

h = histogram(mix_sample, 'Normalization', 'pdf');
alpha(h, 0.4)

% Plot both the Gaussian pdf with realtive weight:
pdf_1_lda = makedist('Normal', mu1_lda, sqrt(sigma_lda));
pdf_0_lda = makedist('Normal', mu0_lda, sqrt(sigma_lda));

x = linspace(mean_mixture - 3*sqrt(sigma2_mixture),...
                mean_mixture + 3*sqrt(sigma2_mixture), granularity);

y1 = w1_lda*pdf(pdf_1_lda, x);
y2 = w0_lda*pdf(pdf_0_lda, x);

y_mixed = y1 + y2;

plot(x, y1, 'LineWidth', 2)
plot(x, y2, 'LineWidth', 2)
plot(x, y_mixed, 'LineWidth', 2)

thresh_mle = thresh_f(mix_sample, pdf_1_lda, pdf_0_lda, ...
                        mu1_lda, mu0_lda, w1_lda, w0_lda);  

l_y = ylim;
line([thresh_mle, thresh_mle], l_y, 'Color', 'r', 'LineWidth', 2)
line([threshold, threshold], l_y, 'Color', 'b', 'LineWidth', 2)

legend('Sample - MIXED', 'MLE pdf - ILL', 'MLE pdf - HEALTHY', ...
            'MLE pdf - MIXED')

hold off

% LDA classifier:

delta_1 = @(x) x.*(mu1_lda/sigma_lda) - ...
                    1/2*mu1_lda^2/sigma_lda + log(w1_lda);
                                    
delta_0 = @(x) x.*(mu0_lda/sigma_lda) - ...
                    1/2*mu0_lda^2/sigma_lda + log(w0_lda);
                                    
LDA_f = (delta_1(mix_sample) > delta_0(mix_sample));

LDA_crit = sigma_lda/(mu0_lda-mu1_lda) * ...
            ((mu0_lda^2-mu1_lda^2)/(2*sigma_lda) + ...
               log((1-w1_lda)/w1_lda));
           
% Errors:

LDA_error = sum(LDA_f ~= sample_division);

LDA_typ1 = mix_sample(LDA_f == 0 & sample_division == 1);
LDA_typ2 = mix_sample(LDA_f == 1 & sample_division == 0);

LDA_emp_bayes_risk = LDA_error/n;

disp(' ')
disp(' LDA Classifier:')
disp(' ')
disp([' LDA criterion:   ', num2str(LDA_crit)])
disp([' LDA error:   ', num2str(LDA_error)])
disp([' LDA empirical risk:   ', num2str(LDA_emp_bayes_risk)])
disp(' ')


figure(5)
hold on

h = histogram(mix_sample, 'Normalization', 'countdensity');
alpha(h, 0.4)

h1 = histogram(LDA_typ1, 'Normalization', 'countdensity');
alpha(h1, 0.7)
h2 = histogram(LDA_typ2, 'Normalization', 'countdensity');
alpha(h2, 0.7)
xlim([mean_mixture - 3*sqrt(sigma2_mixture),...
                mean_mixture + 3*sqrt(sigma2_mixture)]);

l_y = ylim;
line([LDA_crit, LDA_crit], l_y, 'Color', 'r', 'LineWidth', 2)
line([threshold, threshold], l_y, 'Color', 'b', 'LineWidth', 2)

legend('Sample - MIXED', 'LDA - Class=1', 'LDA - Class=0', ...
       'LDA critical v.', 'Bayes critical v.', 'Location', 'best')

hold off


% QDA classifier:
% QDA poses no constraint on the values for sigma12 and sigma02:

% QDA discriminants:

delta_1_q = @(x) -1/2*log(sigma12_s) - 1/2*(x-mu1_lda).^2/sigma12_s ...
                                        + log(w1_lda);
                                    
delta_0_q = @(x) -1/2*log(sigma02_s) - 1/2*(x-mu0_lda).^2/sigma02_s ...
                                        + log(w0_lda);
                                    
QDA_f = (delta_1_q(mix_sample) > delta_0_q(mix_sample));

% QDA_crit = 0; % TODO

% Errors:

QDA_error = sum(QDA_f ~= sample_division);

QDA_typ1 = mix_sample(QDA_f == 0 & sample_division == 1);
QDA_typ2 = mix_sample(QDA_f == 1 & sample_division == 0);

QDA_emp_bayes_risk = QDA_error/n;

disp(' ')
disp(' QDA Classifier:')
disp(' ')
% disp([' QDA criterion:   ', num2str(QDA_crit)])
disp([' QDA error:   ', num2str(QDA_error)])
disp([' QDA empirical risk:   ', num2str(QDA_emp_bayes_risk)])
disp(' ')


figure(6)
hold on

h = histogram(mix_sample, 'Normalization', 'countdensity');
alpha(h, 0.4)

h1 = histogram(QDA_typ1, 'Normalization', 'countdensity');
alpha(h1, 0.7)
h2 = histogram(QDA_typ2, 'Normalization', 'countdensity');
alpha(h2, 0.7)
xlim([mean_mixture - 3*sqrt(sigma2_mixture),...
                mean_mixture + 3*sqrt(sigma2_mixture)]);

l_y = ylim;
% line([QDA_crit, QDA_crit], l_y, 'Color', 'r', 'LineWidth', 2)
line([threshold, threshold], l_y, 'Color', 'b', 'LineWidth', 2)

legend('Sample - MIXED', 'QDA - Class=1', 'QDA - Class=0', ...
       'LDA critical v.', 'Bayes critical v.', 'Location', 'best')

hold off



%% 8.- Construct the logistic classifier associated to the previous sample and 
% compute the associated errors and classifier risk.

loglikeLogit = @(beta) -getLogisticLikelihood(mix_sample, sample_division, beta);

x0 = [1, 1/200];

beta = fminsearch(loglikeLogit, x0);

Logit_crit = -beta(1)/beta(2);

Logit_f = (mix_sample < Logit_crit);

% Errors:

Logit_error = sum(Logit_f ~= sample_division);

Logit_typ1 = mix_sample(Logit_f == 0 & sample_division == 1);
Logit_typ2 = mix_sample(Logit_f == 1 & sample_division == 0);

Logit_emp_bayes_risk = Logit_error/n;

disp(' ')
disp(' Logistic Classifier:')
disp(' ')
disp([' Logit criterion:   ', num2str(Logit_crit)])
disp([' Logit error:   ', num2str(Logit_error)])
disp([' Logit empirical risk:   ', num2str(Logit_emp_bayes_risk)])
disp(' ')


%% 9.- Unsupervised learning. 
% Assume no labels are observed anymore. The only sample which is available
% is the whole features sample x but we assume that the underlying
% distribution is a Gaussian mixture with two components. Construct a
% classifier based on fitting such a distribution to the observed features
% using the technique introduced in Ex_1_1_GaussianMix_sol_withEM. 
% You need both to use the MLE and EM techniques. Compute the associated errors 
% and classifier empirical risk

rng(1)

k = 2;
ops = statset('Display','final','MaxIter',1500,'TolFun',1e-5);

GMModel = fitgmdist(mix_sample, k, 'Options', ops);

mu1_fitgm = GMModel.mu(1);
mu0_fitgm = GMModel.mu(2);
sigma12_fitgm = GMModel.Sigma(:, :, 1);
sigma02_fitgm = GMModel.Sigma(:, :, 2);
w1_fitgm = GMModel.ComponentProportion(1);
w0_fitgm = GMModel.ComponentProportion(2);

pdf_1_gm = makedist('Normal', mu1_fitgm, sqrt(sigma12_fitgm));
pdf_0_gm = makedist('Normal', mu0_fitgm, sqrt(sigma02_fitgm));

thresh_gm = thresh_f(mix_sample, pdf_1_gm, pdf_0_gm, ...
                        mu1_fitgm, mu0_fitgm, w1_fitgm, w0_fitgm);
                    
% Type I errors:

if mu0 > mu1
    typ1 = sum(healthy_sample_0 <= thresh_gm);
else 
    typ1 = sum(ill_sample_1 <= thresh_gm);
end

% Type II errors:

if mu0 > mu1
    typ2 = sum(ill_sample_1 >= thresh_gm);
else 
    typ2 = sum(healthy_sample_0 >= thresh_gm);
end

emp_fitgm_risk = (typ1 + typ2)/n;

disp(' ')
disp(' GMModel MLE classifier:')
disp(' ')
disp([' MLE threshold:   ', num2str(thresh_gm)])
disp([' MLE error:   ', num2str(typ1 + typ2)])
disp([' MLE empirical risk:   ', num2str(emp_fitgm_risk)])
disp(' ')


figure(7)
hold on

h = histogram(mix_sample, 'Normalization', 'countdensity');
alpha(h, 0.4)

h1 = histogram(typ1, 'Normalization', 'countdensity');
alpha(h1, 0.7)
h2 = histogram(typ2, 'Normalization', 'countdensity');
alpha(h2, 0.7)

y1 = w1_fitgm*n*pdf(pdf_1_gm, x);
y2 = w0_fitgm*n*pdf(pdf_0_gm, x);

y_mixed = (y1 + y2);

cmap_w = winter(3);
cmap_h = autumn(3);

plot(x, y1, 'LineWidth', 2, 'Color', cmap_h(2,:))
plot(x, y2, 'LineWidth', 2, 'Color', cmap_w(2,:))
plot(x, y_mixed, 'LineWidth', 2, 'Color', 'k', 'LineStyle', ':')

l_y = ylim;
line([thresh_gm, thresh_gm], l_y, 'Color', 'r', 'LineWidth', 2)
line([threshold, threshold], l_y, 'Color', 'b', 'LineWidth', 2)

legend('Sample - MIXED', 'MLE - Class=1', 'MLE - Class=0', ...
       'GMModel MLE critical v.', 'Bayes critical v.', 'Location', 'best')
title('GMModel MLE unsupervised classification')

grid on

hold off


% EM estimation:

precision = 0.000001;

theta_r = [100, 200, 100, 100, 1/2, 1/2];

theta_em = fitMixedGaussianEMmono(mix_sample, theta_r, precision);

mu1_em = theta_em(1);
mu0_em = theta_em(2);
sigma12_em = theta_em(3);
sigma02_em = theta_em(4);
w1_em = theta_em(5);
w0_em = theta_em(6);

pdf_1_em = makedist('Normal', mu1_em, sqrt(sigma12_em));
pdf_0_em = makedist('Normal', mu0_em, sqrt(sigma02_em));

thresh_em = thresh_f(mix_sample, pdf_1_em, pdf_0_em, ...
                                mu1_em, mu0_em, w1_em, w0_em);                  
                    
% Type I errors:

if mu0 > mu1
    typ1 = sum(healthy_sample_0 <= thresh_em);
else 
    typ1 = sum(ill_sample_1 <= thresh_em);
end

% Type II errors:

if mu0 > mu1
    typ2 = sum(ill_sample_1 >= thresh_em);
else 
    typ2 = sum(healthy_sample_0 >= thresh_em);
end

emp_em_risk = (typ1 + typ2)/n;

disp(' ')
disp(' EM classifier:')
disp(' ')
disp([' EM threshold:   ', num2str(thresh_em)])
disp([' EM error:   ', num2str(typ1 + typ2)])
disp([' EM empirical risk:   ', num2str(emp_em_risk)])
disp(' ')


figure(8)
hold on

h = histogram(mix_sample, 'Normalization', 'countdensity');
alpha(h, 0.4)

h1 = histogram(typ1, 'Normalization', 'countdensity');
alpha(h1, 0.7)
h2 = histogram(typ2, 'Normalization', 'countdensity');
alpha(h2, 0.7)

y1 = w1_em*n*pdf(pdf_1_em, x);
y2 = w0_em*n*pdf(pdf_0_em, x);

y_mixed = (y1 + y2);

cmap_w = winter(3);
cmap_h = autumn(3);

plot(x, y1, 'LineWidth', 2, 'Color', cmap_h(2,:))
plot(x, y2, 'LineWidth', 2, 'Color', cmap_w(2,:))
plot(x, y_mixed, 'LineWidth', 2, 'Color', 'k', 'LineStyle', ':')

l_y = ylim;
line([thresh_em, thresh_em], l_y, 'Color', 'r', 'LineWidth', 2)
line([threshold, threshold], l_y, 'Color', 'b', 'LineWidth', 2)

legend('Sample - MIXED', 'EM - Class=1', 'EM - Class=0', ...
       'EM critical v.', 'Bayes critical v.', 'Location', 'best')
title('EM unsupervised classification')

grid on

hold off

