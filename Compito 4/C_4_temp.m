% -------------------------------------------
%
%   Compito 4 - TAE
%
% -------------------------------------------

clear

% Normal PDF for sample:
% x, y, b1, b2, se
loglike = @(z) log(1./(2*pi*z(:,5)) .* ...
                exp(-(z(:,2) - z(:,3) - z(:,1).^z(:,4)).^2 ./ (2*z(:,5))));
            
syms x y b1 b2 se
% logl(x,y,b1,b2,se) = log(1./(2*pi*se) * ...
%                         exp(-((y - b1 - x.^b2)'*(y - b1 - x.^b2))/(2*se)));
logl(x,y,b2,se) = -0.5*log(se) - ...
                        ((y - x.^b2)'*(y - x.^b2))/(2*se);

grad_logl = gradient(logl, [b2, se]);
hess_logl = hessian(logl, [b2, se]);

% Parameters:

beta1 = 1;
beta2 = 1/10;
sigma_e = 0.1;

n = 1000;

% Sample generation:

X = [ones(n, 1), repmat(transpose(1:10), 100, 1)];
                
%% Newton-Rapson:

for j = 1:1

    Y = transpose([beta1, 1, 1] * [X(:,1), X(:,2).^beta2, normrnd(0,sqrt(sigma_e),[n,1])]');
    
    theta0 = [2, 1, 2];
    
    diff = 1;
    it = 0;
    
    while (diff > 10^-12 && it < 5)
    
        tic
        
        llike = @(th)  n*0.5*log(th(3)^2) + ...
                 ((Y - X(:,1)*th(1) - X(:,2).^th(2))'*...
                           (Y - X(:,1)*th(1) - X(:,2).^th(2))) ./ (2*th(3)^2);
        
        M = fminunc(llike, [0.8, 0.2, 0.15]);
            
%         x_ = -10:0.1:10;
%         y_ = 0.1:0.1:10;
%         
%         [xx, yy] = meshgrid(x_,y_);
%         
%         xx_ = xx(:);
%         yy_ = yy(:);
%         
%         for l = 1:length(xx(:))
%             z_(l) = llike([xx_(l); yy_(l)]);
%         end
%         
%         zz = reshape(z_, size(xx));
%             
%         D1_ll = gradest(llike, theta0);    
%         H_ll = hessian(llike, theta0);
%         
%         D1_sym = zeros(3,1);
%         H_sym = zeros(3);
%         
%         for k = 1:100
%             D1_sym = D1_sym + ...
%                 double(grad_logl(Y(k), X(k), theta0(1), theta0(2), theta0(3)));
%             H_sym = H_sym + ...
%                 double(hess_logl(Y(k), X(k), theta0(1), theta0(2), theta0(3)));
%         end
        
        toc

%         theta = transpose(theta0(:) - (H_ll \ D1_ll(:)));
%         thetasym = transpose(theta0(:) - (H_sym^-1 * D1_sym(:)));
%         
%         diff = norm(thetasym - theta0);
%         theta0 = thetasym;
        
        it = it + 1;
       
    end
    
end