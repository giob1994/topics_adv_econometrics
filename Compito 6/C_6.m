% -------------------------------------------
%
%   Compito 6 - TAE
%
% -------------------------------------------

clear

% Normal OLS log-likelihood for sample:
%  z = [ x, y, beta1, sigma ]
loglike = @(z) -0.5*log(z(4)) - 1/(2*z(4)) * (z(2) - z(1)*z(3))^2;

syms x y b s
logl(x,y,b,s) = -0.5*log(s) - 1/(2*s) * (y - x*b)^2;

pd_b_1 = diff(logl, b);
pd_b_2 = diff(pd_b_1, b);
pd_s_1 = diff(logl, s);
pd_s_2 = diff(pd_s_1, s);
pd_bs = diff(pd_b_1, s);

% Set up the coefficients:

beta1 = 1;
sigma = 1;

n = 1000;

MC_size = 1;

X = repmat(transpose(1:10), 100, 1);
X(500) = 10^(2.001);
X(1000) = 10^(2.001);

OLS_beta1 = zeros(MC_size, 1);

for j = 1:MC_size
    
     Y = transpose([beta1, 1] * [X, sqrt(sigma).*randn(n,1)]');
     
     % Approximate MLE with OLS estimates:
     OLS_beta1(j, :) = (X'*X) \ (X'*Y);
     
     res = Y - OLS_beta1(j, :)*X;
     OLS_sigma = (res'*res) / (n - 1);
     
     % Log-likelihood Hessian estimation:
     H = zeros(2);
     Hsym = zeros(2);
     for k = 1:n
         
         % Numerical Hessian:
         H_ = hessian(loglike, [X(k), Y(k), OLS_beta1(j, :), OLS_sigma]);
         
         % Analytical Hessian:
         H_sym = [pd_b_2(X(k), Y(k), OLS_beta1(j, :), OLS_sigma), ...
                  pd_bs(X(k), Y(k), OLS_beta1(j, :), OLS_sigma); ...
                  pd_bs(X(k), Y(k), OLS_beta1(j, :), OLS_sigma), ...
                  pd_s_2(X(k), Y(k), OLS_beta1(j, :), OLS_sigma)];
         
         H = H + H_(3:end,3:end);
         
         Hsym = Hsym + double(H_sym);
     end
    
     % t-ratios:
     
     1;
    
end 
