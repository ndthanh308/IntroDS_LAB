function out = lie_brackets(f,g,var)



out = jacobian(g,var) * f - jacobian(f, var) * g;

end
