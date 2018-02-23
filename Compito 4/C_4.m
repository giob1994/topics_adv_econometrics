% -------------------------------------------
%
%   Compito 4 - TAE
%
% -------------------------------------------

clear

% Normal PDF for sample:
% x, y, b1, b2, se
loglike = @(z) log(1./(2*pi*z(:,5)^2) .* ...
                exp(-(z(:,2) - z(:,3) - z(:,1).^z(:,4)).^2 ./ (2*z(:,5)^2)));
            
syms x y b1 b2 se
% logl(x,y,b1,b2,se) = log(1./(2*pi*se^2) * exp(-(y - b1 - x^b2)^2 / (2*se^2)));
logl(x,y,b1,b2,se) = -0.5*log(se^2) - ...
                        ((y - b1 - x.^b2)'*(y - b1 - x.^b2))/(2*se^2);

pd_b1_1 = diff(logl, b1);
pd_b1_2 = diff(pd_b1_1, b1);
pd_b2_1 = diff(logl, b2);
pd_b2_2 = diff(pd_b2_1, b2);
pd_se_1 = diff(logl, se);
pd_se_2 = diff(pd_se_1, se);
pd_b1se = diff(pd_b1_1, se);
pd_b2se = diff(pd_b2_1, se);
pd_b1b2 = diff(pd_b1_1, b2);

% Parameters:

beta1 = 1;
beta2 = 1/10;
sigma_e = 0.1;

n = 1000;

% Sample generation:

X = [ones(n, 1), repmat(transpose(1:10), 100, 1)];
                
%% Newton-Rapson:

for j = 1:1

    Y = transpose([beta1, beta2, 1] * [X, sqrt(sigma_e).*randn(n,1)]');
    
    theta0 = [1, 1/10, 0.1];
    theta0_ll = theta0;
    
    diff = 1;
    it = 0;
    
    while (diff > 10^-12 && it < 5)
    
        D1 = zeros(1,3);
        H = zeros(3);
        
%         D1sym = zeros(1,3);
%         Hsym = zeros(3);

        tic
        for k = 1:100
            
            % Numerical:
            D1_ = gradest(loglike, [X(k,2), Y(k), theta0]);
            D1 = D1 + D1_(3:end);

            H_ = hessian(loglike, [X(k,2), Y(k), theta0]);
            H = H + H_(3:end,3:end);
            
            % Symbolic:
%             D1sym = D1sym + ...
%                     double([pd_b1_1(X(k,2), Y(k), theta0(1), theta0(2), theta0(3)), ...
%                             pd_b2_1(X(k,2), Y(k), theta0(1), theta0(2), theta0(3)), ...
%                             pd_se_1(X(k,2), Y(k), theta0(1), theta0(2), theta0(3))]);
%                         
%             Hsym = Hsym + ...
%                    double([pd_b1_2(X(k,2), Y(k), theta0(1), theta0(2), theta0(3)), ...
%                             pd_b1b2(X(k,2), Y(k), theta0(1), theta0(2), theta0(3)), ...
%                             pd_b1se(X(k,2), Y(k), theta0(1), theta0(2), theta0(3)); 
%                             pd_b1b2(X(k,2), Y(k), theta0(1), theta0(2), theta0(3)), ...
%                             pd_b2_2(X(k,2), Y(k), theta0(1), theta0(2), theta0(3)), ...
%                             pd_b2se(X(k,2), Y(k), theta0(1), theta0(2), theta0(3));
%                             pd_b1se(X(k,2), Y(k), theta0(1), theta0(2), theta0(3)), ...
%                             pd_b2se(X(k,2), Y(k), theta0(1), theta0(2), theta0(3)), ...
%                             pd_se_2(X(k,2), Y(k), theta0(1), theta0(2), theta0(3));]);
        end
        
%         llike = @(th) log( 1./(2*pi*th(3)) .* ...
%                 exp(- ((Y - X*th(1:2)')'*(Y - X*th(1:2)')) ./ (2*th(3))));
%             
%         D1_ll = gradest(llike, theta0);    
%         H_ll = hessian(llike, theta0);
        
        toc

        theta = transpose(theta0(:) - (H \ D1(:)));
%         theta_ll = transpose(theta0_ll(:) - (H_ll \ D1_ll(:)));
%         thetasym = transpose(theta0(:) - (Hsym \ D1sym(:)));
        
        diff = norm(theta - theta0);
        theta0 = theta;
%         theta0_ll = theta_ll;
        
        it = it + 1;
       
    end
    
end