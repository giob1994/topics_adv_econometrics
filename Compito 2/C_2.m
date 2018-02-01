% -------------------------------------------
%
%   Compito 2 - TAE
%
% -------------------------------------------

clear

% Set up the coefficients:

beta1 = 1;
beta2 = 0.1;

MC_size = 10000;

x = [1, 3, 2, 1];
X = [ones(size(x)); x];

%% MC replication:

W = [0, 2; 1, 1; 4, 1; 1, 0];

OLS_betas    = zeros(MC_size, 2);
LinEst_betas = zeros(MC_size, 2);

for j = 1:MC_size
    
   tmp_Y = [beta1, beta2, 1] * [X; randn(size(x))];
   
   OLS_betas(j,:)    = (X*X') \ (X*tmp_Y');
   LinEst_betas(j,:) = (W'*X') \ (W'*tmp_Y');
    
end

%% Results:

res_OLS = mean(OLS_betas);
res_LinEst = mean(LinEst_betas);

OLS_var_theoretical = inv(X*X');

OLS_var_emp = cov(OLS_betas);
LinEst_var_emp = cov(LinEst_betas);

