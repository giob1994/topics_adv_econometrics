function [c, ceq] = confuneq(x)
    % Nonlinear inequality constraints
    Sigma_0 = math([x(5), x(6), x(7)]);
    Sigma_1 = math([x(8), x(9), x(10)]);
    eigS0 = -min(eigs(Sigma_0));
    eigS1 = -min(eigs(Sigma_1));
    c(1) = eigS0 + 0.00001;
    c(2) = eigS1 + 0.00001;
    ceq = [];
