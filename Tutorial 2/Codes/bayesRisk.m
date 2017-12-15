%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear 
close all
rng default;

p_mixture = 1/3;

w1 = p_mixture;
w2 = 1 - p_mixture;

mu1 = 100;
sigma12 = 900;

mu2 = 150;
sigma22 = 550;

% Gaussian distributions:
pdf_1 = @(x) 1/sqrt(2*pi*sigma12).*exp(-((x-mu1).^2)./(2*sigma12));
pdf_2 = @(x) 1/sqrt(2*pi*sigma22).*exp(-((x-mu2).^2)./(2*sigma22));

f = @(x) w1.*pdf_1(x) - w2.*pdf_2(x);
% f = @(x) pdf_1(x) - pdf_2(x);

bound = fzero(f, max([(mu1-mu2), (mu2-mu1)]));

lb = min([mu1,mu2])-5*sqrt(max([sigma12,sigma22]));
ub = max([mu1,mu2])+5*sqrt(max([sigma12,sigma22]));

% risk = w2*integral(pdf_2, lb, bound) + w1*integral(pdf_1, bound, ub);

risk = w2*cdf('Normal', bound, mu2, sqrt(sigma22)) + ...
       w1*(1-cdf('Normal', bound, mu1, sqrt(sigma12)));


% eta = @(x) 1-(w1.*pdf_1(x))./(w1.*pdf_1(x)+w2.*pdf_2(x));
% risk = integral(eta, -100, boundary);


% Plot:

lin = linspace(lb, ub, 2000);

figure(1)
hold on

plot(lin, w1.*pdf_1(lin))
plot(lin, w2.*pdf_2(lin))
line([bound, bound], [0, 0.02], 'Color', 'k')
legend('pdf 1', 'pdf 2')

