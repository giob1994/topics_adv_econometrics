% The objective of this exercise is simulating out of a Gaussian mixture
% model and fitting a Gaussian mixture distribution to a
% data set by maximizing the log-likelihood, using Expectation-Maximization
% (EM)  algorithm and built-in matlab function with implemented EM

% We consider first the case of mixture of one-dimensional Gaussian distributions.
% By the next tutorial you need to extend the codes for the case of
% multivariate Gaussian mixtures with arbitrary number of components.

clear 
close all
rng default;

% 11.-Extend the codes to the case of bivariate Gaussian till the next
% tutorial

% Multivariate gaussian distributions:

n = 1000;
p_mixture = 1/3;

w1 = p_mixture;
w2 = 1 - p_mixture;

mu1 = [2.3, 1];
sigma12 = [1, 0.2; 0.2, 1.3];

mu2 = [-1, 0.1];
sigma22 = [1.3, 0; 0, 1.5];

bern = binornd(1, p_mixture, n, 1);

n_component1 = sum(bern);
n_component2 = n - n_component1;

gauss_sample_1 = mvnrnd(mu1, sigma12, n_component1);
gauss_sample_2 = mvnrnd(mu2, sigma22, n_component2);

Gauss_mix_sample = [gauss_sample_1; gauss_sample_2];


% Plot the 2D distribution of the sample:

figure(1)

subplot(1,2,1)
hold on
title('Separate samples')
plot(gauss_sample_1(:,1), gauss_sample_1(:,2), 'b+')
plot(gauss_sample_2(:,1), gauss_sample_2(:,2), 'ro')
grid on
pbaspect([2 2 1])
hold off

subplot(1,2,2)
hold on
title('Mixed sample')
plot(Gauss_mix_sample(:,1), Gauss_mix_sample(:,2), 'kx')
grid on
pbaspect([2 2 1])
hold off


% Plot in 3D the histogram of the sample, the theoretical mixed 
% distribution and also the two original (weigthed) distributions:

figure(2)
ax2 = subplot(1,2,1);
hold on

granularity = 100;

h = histogram2(Gauss_mix_sample(:,1), Gauss_mix_sample(:,2),...
    floor(granularity/3), 'Normalization', 'pdf', 'FaceAlpha', 0.2);

% Plot both the Gaussian pdf with relative weight:
mean_mixture = w1*mu1 + w2*mu2;
sigma2_mixture = w1*((mu1(:) - mean_mixture(:))*(mu1(:) - mean_mixture(:))'+sigma12) + ...
                    w2*((mu2(:) - mean_mixture(:))*(mu2(:) - mean_mixture(:))'+sigma22);
                
x1 = linspace(min(Gauss_mix_sample(:,1))-1, ...
                max(Gauss_mix_sample(:,1))+1, granularity/3);
x2 = linspace(min(Gauss_mix_sample(:,2))-1, ...
                max(Gauss_mix_sample(:,2))+1, granularity/3);
[X1, X2] = meshgrid(x1,x2);

y1 = w1*reshape(mvnpdf([X1(:), X2(:)], mu1, sigma12), length(x1), length(x2));
y2 = w2*reshape(mvnpdf([X1(:), X2(:)], mu2, sigma22), length(x1), length(x2));

y_mixed = y1 + y2;
                
surf(x1, x2, y_mixed, 'FaceAlpha', 0.7);
shading interp
colormap jet
% mesh(x1, x2, y_mixed);
pbaspect([1 1 0.5])
xlim([-6, 6]);
ylim([-6, 6]);
view(3)
grid on

title('Sample histogram - Mixed PDF')

hold off

% figure(3)
ax1 = subplot(1,2,2);
hold on

s1 = surf(x1, x2, y1, 'FaceAlpha', 0.5);
s1 = surf(x1, x2, y2, 'FaceAlpha', 0.5);
colormap jet
% shading flat
pbaspect([1 1 0.5])
xlim([-6, 6]);
ylim([-6, 6]);
view(3)
grid on

title('Weigthed PDFs')

hold off

hlink = linkprop([ax1,ax2],{'CameraPosition','CameraUpVector'});
addprop(hlink,'PlotBoxAspectRatio')
rotate3d on


% Use Maximum Likelihood estimation to fit the sample:

mus = [mu1; mu2];
sigmas = zeros([size(sigma12), 2]);
sigmas(:,:,1) = sigma12;
sigmas(:,:,2) = sigma22;

theta = struct('n', 2,...
               'mu', mus,...
               'sigma', sigmas,...
               'w', [w1, w2]);
           
% getMultiGaussianMixLogLikelihood(Gauss_mix_sample, theta)

loglikeF = @(th) -getMultiGaussianMixLogLikelihood(Gauss_mix_sample, th);

x0 = [2,...
      0, 0, 0, 0,...
      1, 0, 0, 1, 1, 0, 0, 1,...
      1/2, 1/2];
Aeq = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1];
beq = 1;
lb = [2,...
      -Inf, -Inf, -Inf, -Inf,...
      0, 0, 0, 0, 0, 0, 0, 0,...
      0, 0];
ub = [2,...
      Inf, Inf, Inf, Inf,...
      Inf, Inf, Inf, Inf, Inf, Inf, Inf, Inf,...
      1, 1];
options = optimoptions(@fmincon, 'Display', 'iter');

% [theta_mle, log_like_mle] = ...
%     fmincon(loglikeF, x0, [], [], Aeq, beq, lb, ub, [], options);

% MLE using the built-in fitting Toolbox:

k = 2;
ops = statset('Display','final','MaxIter',1500,'TolFun',1e-5);

GMModel = fitgmdist(Gauss_mix_sample, k, 'Options', ops);

mu1_fitgm = GMModel.mu(1);
mu2_fitgm = GMModel.mu(2);
sigma12_fitgm = GMModel.Sigma(:, :, 1);
sigma22_fitgm = GMModel.Sigma(:, :, 2);
w1_fitgm = GMModel.ComponentProportion(1);
w2_fitgm = GMModel.ComponentProportion(2);

figure(3)
hold on

granularity = 100;

h = histogram2(Gauss_mix_sample(:,1), Gauss_mix_sample(:,2),...
    floor(granularity/3), 'Normalization', 'pdf', 'FaceAlpha', 0.2);

% Plot:
                
x1 = linspace(min(Gauss_mix_sample(:,1))-1, ...
                max(Gauss_mix_sample(:,1))+1, granularity/3);
x2 = linspace(min(Gauss_mix_sample(:,2))-1, ...
                max(Gauss_mix_sample(:,2))+1, granularity/3);
[X1, X2] = meshgrid(x1,x2);

y1 = w1_fitgm*reshape(mvnpdf([X1(:), X2(:)],...
    mu1_fitgm, sigma12_fitgm), length(x1), length(x2));
y2 = w2_fitgm*reshape(mvnpdf([X1(:), X2(:)],...
    mu2_fitgm, sigma22_fitgm), length(x1), length(x2));

y_mixed = y1 + y2;
                
surf(x1, x2, y_mixed, 'FaceAlpha', 0.7);
shading interp
colormap jet
pbaspect([1 1 0.5])
xlim([-6, 6]);
ylim([-6, 6]);
view(3)
grid on
rotate3d on

title('MLE estimation')

hold off

% EM algorithm for multidimensional case:


