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
xlim([-6, 6])
ylim([-6, 6])
xlim
view(3)
grid on

hold off

% figure(3)
ax1 = subplot(1,2,2);
hold on

s1 = surf(x1, x2, y1, 'FaceAlpha', 0.5);
s1 = surf(x1, x2, y2, 'FaceAlpha', 0.5);
colormap jet
% shading flat
pbaspect([1 1 0.5])
xlim([-6, 6])
ylim([-6, 6])
view(3)
grid on

hold off

hlink = linkprop([ax1,ax2],{'CameraPosition','CameraUpVector'});
addprop(hlink,'PlotBoxAspectRatio')
rotate3d on




