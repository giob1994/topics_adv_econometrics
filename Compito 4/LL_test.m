clear

% Parameters:

beta1 = 1;
beta2 = 1/10;
sigma_e = 0.01;


% Log-likelihood:

logl = @(x,y,b1,b2,se) -0.5*log(se^2) - ...
                        ((y - b1 - x*b2)'*(y - b1 - x*b2))/(2*se^2);
like = @(x,y,b1,b2,se) 1/sqrt(2*pi*se^2) .* ...
                    exp( -((y - b1 - x*b2)'*(y - b1 - x*b2))/(2*se^2) );
                                  

% Sample generation:

n = 1000;
X = repmat(transpose(1:10), 100, 1);
Y = transpose([beta1, beta2, 1] * [ones(n,1), X, sqrt(sigma_e).*randn(n,1)]');

% Plot:

x_ = -2:0.02:2;
y_ = 0.1:0.1:20;

[xx, yy] = meshgrid(x_,y_);

xx_ = xx(:);
yy_ = yy(:);
for l = 1:length(xx(:))
%     ll(l) = logl(X, Y, 1, xx_(l), yy_(l));
    ll(l) = like(X, Y, xx_(l), 0.1, yy_(l));
end
ll = reshape(ll, size(xx));

m = mesh(xx, yy, ll);
ax = gca;
% ax.ZAxis.Scale = 'log';

% % MLE:
% L = 
% 
% theta0 = [2, 1, 2];
% 
% A = [0, 0, -1];
% b = 0;
% 
% M = fmincon(L, [0.5, 0.5, 0.5], A, b);