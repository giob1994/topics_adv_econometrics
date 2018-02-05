function [grad] = grad_num(f, x0)

n = length(x0);

delta = 10^-2;

prec = 10^-8;

grad0 = ones(size(x0));
grad = zeros(size(x0));

diff = 1;
it = 0;

while (diff > prec && it < 100)
    
    f1 = f(delta*eye(length(x0)) + repmat(x0,n,1));
    f2 = f(-delta*eye(length(x0)) + repmat(x0,n,1));
    
    grad = (f1 - f2) / delta;
    
    diff = norm(grad - grad0);
    
    grad0 = grad;
    delta = 0.5 * delta;
    
    it = it+1;
    
end

grad = grad / 2;

end

