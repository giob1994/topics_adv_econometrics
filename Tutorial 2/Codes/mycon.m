function [ c, ceq ] = mycon( x )

c = 0;

sigma_con = [0, 0, -1, 1, 0, 0];
ceq = sigma_con * x(:); 

end

