function ops = thefunctions()
ops.rosenbrockFunction = @rosenbrockFunction;
ops.rosenbrockGradient = @rosenbrockGradient;
ops.rosenbrockHessian = @rosenbrockHessian;
ops.rosenbrockResiduals = @rosenbrockResiduals;
ops.rosenbrockResidualJacobian = @rosenbrockResidualJacobian;

    function F = rosenbrockFunction(x, y)
        F = 100 * (y - x.^2).^2 + (1 - x).^2;
    end

    function grad = rosenbrockGradient(x, y)
        grad = [2*(x - 1) - 400*x.*(y - x.^2); 200*(y - x.^2)];
    end

    function hessian = rosenbrockHessian(x, y)
        hessian = [2 - 400*y + 1200*x.^2, -400*x; -400*x, 200];
    end

    function residuals = rosenbrockResiduals(x, y)
        residuals = [10 * (y - x.^2); 1 - x];
    end

    function jacobian = rosenbrockResidualJacobian(x, y)
        jacobian = [-20*x, 10; -1, 0];
    end
end
