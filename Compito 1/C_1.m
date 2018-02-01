% -------------------------------------------
%
%   Compito 1 - TAE
%
% -------------------------------------------

clear

% Set up the coefficients:

beta1 = 1;
beta2 = 0.1;

MC_size = 1000;

% Create the AR(1) model:

ar_model = arima('AR', beta2, 'Constant', beta1, ...
                    'Distribution', 'Gaussian', 'Variance', 1);

%% (A) Sample size T = 4:

OLS_betas = zeros(MC_size, 2);

for j = 1:MC_size

    [sim_, ~] = simulate(ar_model, 4, 'Y0', 1);
    
    mdl = fitlm(sim_(2:end), sim_(1:end-1));
    
    OLS_betas(j, :) = transpose(mdl.Coefficients.Estimate);
    
end

disp(' ')
disp('  Completed M.C. simulation with T = 4')

res1 = mean(OLS_betas);


%% (B) Sample size T = 20:

OLS_betas = zeros(MC_size, 2);

for j = 1:MC_size

    [sim_, ~] = simulate(ar_model, 20, 'Y0', 1);
    
    mdl = fitlm(sim_(2:end), sim_(1:end-1));
    
    OLS_betas(j, :) = transpose(mdl.Coefficients.Estimate);
    
end

disp(' ')
disp('  Completed M.C. simulation with T = 20')

res2 = mean(OLS_betas);


%% (C) Sample size T = 20:

OLS_betas = zeros(MC_size, 2);

for j = 1:MC_size

    [sim_, ~] = simulate(ar_model, 100, 'Y0', 1);
    
    mdl = fitlm(sim_(2:end), sim_(1:end-1));
    
    OLS_betas(j, :) = transpose(mdl.Coefficients.Estimate);
    
end

disp(' ')
disp('  Completed M.C. simulation with T = 100')

res3 = mean(OLS_betas);


%% Print biases:

disp(' ')
disp([' Biases with T = 4:    ', num2str(res1-[beta1, beta2])]);

disp(' ')
disp([' Biases with T = 20:   ', num2str(res2-[beta1, beta2])]);

disp(' ')
disp([' Biases with T = 100:  ', num2str(res3-[beta1, beta2])]);

