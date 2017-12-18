function [ c, ceq ] = defposconst( x )

c = 0;

dims = 2;

Sigma0  = [ x(3+2*dims), x(4+2*dims);
            x(4+2*dims), x(5+2*dims); ];
Sigma1  = [ x(3+2*dims+dims^2-1), x(4+2*dims+dims^2-1);
            x(4+2*dims+dims^2-1), x(5+2*dims+dims^2-1); ];

[~,p0] = chol(Sigma0);
[~,p1] = chol(Sigma1);

ceq = p0 + p1;

end
