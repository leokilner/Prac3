%% Task 1
clear all
clc

% Define the Rosenbrock function and its derivatives
x = linspace(-2, 2, 201);
y = linspace(-2, 2, 201);
[X, Y] = meshgrid(x, y);
ops = thefunctions();
F = ops.rosenbrockFunction(X, Y);

rgbAnchors = [0 0 1; 0 1 1; 0 1 0; 1 1 0; 1 0 0];
rgbScale = interp1(linspace(0, 1, size(rgbAnchors, 1)), rgbAnchors, ...
    linspace(0, 1, 256));

figure;
meshIndices = 1:10:numel(x);
hSurface = surf(X(meshIndices, meshIndices), Y(meshIndices, meshIndices), ...
    F(meshIndices, meshIndices), 'EdgeColor', [0.83 0.83 0.83], ...
    'LineWidth', 0.1, 'DisplayName', 'f(x,y)');
hold on;
hMinimum = scatter3(-1, 1, 0, 45, [1 0 1], 'filled', ...
    'MarkerEdgeColor', 'k', 'DisplayName', 'Minimum (-1, 1, 0)');
hold off;
colormap(rgbScale);
caxis([0 max(F(:))]);
colorbar;
xlabel('x');
ylabel('y');
zlabel('f(x,y)');
title('3D Surface Plot of f(x,y)');
legend('Location', 'northeast');
grid on;
view(3);

figure;
[~, hContours] = contourf(X, Y, F, 40, 'LineColor', 'none');
hContours.DisplayName = 'f(x,y) contours';
hold on;
hMinimum2D = plot(-1, 1, 'o', 'MarkerSize', 7 / sqrt(2), ...
    'MarkerFaceColor', [1 0 1], 'MarkerEdgeColor', 'k', ...
    'DisplayName', 'Minimum (-1, 1)');
hold off;
colormap(rgbScale);
caxis([0 max(F(:))]);
colorbar;
xlabel('x');
ylabel('y');
title('2D Filled Contour Plot of f(x,y)');
legend('Location', 'northeast');
axis tight;
grid on;

%% Task 2

% Random initialization of the starting point
x0 = rand() * 4 - 2;
y0 = rand() * 4 - 2;

% Finding the minimum using gradient descent
gdPoint = [x0; y0];
gdTolerance = 1e-6;
gdMaxIterations = 100000;
gdArmijo = 1e-4;
gdBacktrackFactor = 0.5;
gdMaxLineSearchIterations = 50;
gdPath = zeros(2, gdMaxIterations + 1);
gdPath(:, 1) = gdPoint;
gdIteration = 0;
gdConverged = false;

while gdIteration < gdMaxIterations
   gdGradient = ops.rosenbrockGradient(gdPoint(1), gdPoint(2));
   gdGradientNorm = norm(gdGradient);
   if gdGradientNorm <= gdTolerance
       gdConverged = true;
       break;
   end

   gdValue = ops.rosenbrockFunction(gdPoint(1), gdPoint(2));
   gdStepSize = 1;
   gdAccepted = false;
   for gdLineSearchIteration = 1:gdMaxLineSearchIterations
       gdTrialPoint = gdPoint - gdStepSize * gdGradient;
       gdTrialValue = ops.rosenbrockFunction(gdTrialPoint(1), gdTrialPoint(2));
       if gdTrialValue <= gdValue - gdArmijo * gdStepSize * (gdGradient' * gdGradient)
           gdAccepted = true;
           break;
       end
       gdStepSize = gdBacktrackFactor * gdStepSize;
   end

   if ~gdAccepted
       error('GradientDescent:LineSearchFailed', ...
           'Gradient descent could not find a decreasing step.');
   end

   gdPoint = gdTrialPoint;
   gdIteration = gdIteration + 1;
   gdPath(:, gdIteration + 1) = gdPoint;
end

gdGradient = ops.rosenbrockGradient(gdPoint(1), gdPoint(2));
gdGradientNorm = norm(gdGradient);
gdValue = ops.rosenbrockFunction(gdPoint(1), gdPoint(2));
gdConverged = gdConverged || gdGradientNorm <= gdTolerance;
gdPath = gdPath(:, 1:gdIteration + 1);

if ~gdConverged
   warning('GradientDescent:NoConvergence', ...
       'Gradient descent did not converge within %d iterations.', gdMaxIterations);
end
fprintf(['Gradient descent: x = %.8g, y = %.8g, f = %.8g, ', ...
   'gradient norm = %.3g, iterations = %d, converged = %d\n'], ...
   gdPoint(1), gdPoint(2), gdValue, gdGradientNorm, gdIteration, gdConverged);

% Finding the minimum using Newton's method
newtonPoint = [x0; y0];
newtonTolerance = 1e-6;
newtonMaxIterations = 1000;
newtonArmijo = 1e-4;
newtonBacktrackFactor = 0.5;
newtonMaxLineSearchIterations = 50;
newtonPath = zeros(2, newtonMaxIterations + 1);
newtonPath(:, 1) = newtonPoint;
newtonIteration = 0;
newtonConverged = false;

while newtonIteration < newtonMaxIterations
   newtonGradient = ops.rosenbrockGradient(newtonPoint(1), newtonPoint(2));
   newtonGradientNorm = norm(newtonGradient);
   if newtonGradientNorm <= newtonTolerance
       newtonConverged = true;
       break;
   end

   newtonHessian = ops.rosenbrockHessian(newtonPoint(1), newtonPoint(2));
   newtonDamping = 0;
   newtonDirectionFound = false;
   for newtonDampingIteration = 1:20
       newtonDampedHessian = newtonHessian + newtonDamping * eye(2);
       [~, newtonCholeskyFlag] = chol(newtonDampedHessian);
       if newtonCholeskyFlag == 0
           newtonDirection = -(newtonDampedHessian \ newtonGradient);
           if all(isfinite(newtonDirection)) && ...
                   newtonGradient' * newtonDirection < 0
               newtonDirectionFound = true;
               break;
           end
       end
       if newtonDamping == 0
           newtonDamping = 1e-6;
       else
           newtonDamping = 10 * newtonDamping;
       end
   end

   if ~newtonDirectionFound
       error('NewtonMethod:DirectionFailed', ...
           'Newton method could not find a descent direction.');
   end

   newtonValue = ops.rosenbrockFunction(newtonPoint(1), newtonPoint(2));
   newtonDirectionalDerivative = newtonGradient' * newtonDirection;
   newtonStepSize = 1;
   newtonAccepted = false;
   for newtonLineSearchIteration = 1:newtonMaxLineSearchIterations
       newtonTrialPoint = newtonPoint + newtonStepSize * newtonDirection;
       newtonTrialValue = ops.rosenbrockFunction( ...
           newtonTrialPoint(1), newtonTrialPoint(2));
       if newtonTrialValue <= newtonValue + ...
               newtonArmijo * newtonStepSize * newtonDirectionalDerivative
           newtonAccepted = true;
           break;
       end
       newtonStepSize = newtonBacktrackFactor * newtonStepSize;
   end

   if ~newtonAccepted
       error('NewtonMethod:LineSearchFailed', ...
           'Newton method could not find a decreasing step.');
   end

   newtonPoint = newtonTrialPoint;
   newtonIteration = newtonIteration + 1;
   newtonPath(:, newtonIteration + 1) = newtonPoint;
end

newtonGradient = ops.rosenbrockGradient(newtonPoint(1), newtonPoint(2));
newtonGradientNorm = norm(newtonGradient);
newtonValue = ops.rosenbrockFunction(newtonPoint(1), newtonPoint(2));
newtonConverged = newtonConverged || newtonGradientNorm <= newtonTolerance;
newtonPath = newtonPath(:, 1:newtonIteration + 1);

if ~newtonConverged
   warning('NewtonMethod:NoConvergence', ...
       'Newton method did not converge within %d iterations.', newtonMaxIterations);
end
fprintf(['Newton method: x = %.8g, y = %.8g, f = %.8g, ', ...
   'gradient norm = %.3g, iterations = %d, converged = %d\n'], ...
   newtonPoint(1), newtonPoint(2), newtonValue, newtonGradientNorm, ...
   newtonIteration, newtonConverged);

% Finding the minimum using the Gauss-Newton method
gnPoint = [x0; y0];
gnTolerance = 1e-6;
gnMaxIterations = 1000;
gnArmijo = 1e-4;
gnBacktrackFactor = 0.5;
gnMaxLineSearchIterations = 50;
gnPath = zeros(2, gnMaxIterations + 1);
gnPath(:, 1) = gnPoint;
gnIteration = 0;
gnConverged = false;

while gnIteration < gnMaxIterations
   gnResiduals = ops.rosenbrockResiduals(gnPoint(1), gnPoint(2));
   gnGradient = ops.rosenbrockGradient(gnPoint(1), gnPoint(2));
   gnGradientNorm = norm(gnGradient);
   if gnGradientNorm <= gnTolerance
       gnConverged = true;
       break;
   end

   gnJacobian = ops.rosenbrockResidualJacobian(gnPoint(1), gnPoint(2));
   gnNormalMatrix = gnJacobian' * gnJacobian;
   gnDirection = -(gnNormalMatrix \ (gnJacobian' * gnResiduals));
   gnDirectionalDerivative = gnGradient' * gnDirection;
   if ~all(isfinite(gnDirection)) || gnDirectionalDerivative >= 0
       error('GaussNewton:DirectionFailed', ...
           'Gauss-Newton method could not find a descent direction.');
   end

   gnValue = ops.rosenbrockFunction(gnPoint(1), gnPoint(2));
   gnStepSize = 1;
   gnAccepted = false;
   for gnLineSearchIteration = 1:gnMaxLineSearchIterations
       gnTrialPoint = gnPoint + gnStepSize * gnDirection;
       gnTrialValue = ops.rosenbrockFunction(gnTrialPoint(1), gnTrialPoint(2));
       if gnTrialValue <= gnValue + ...
               gnArmijo * gnStepSize * gnDirectionalDerivative
           gnAccepted = true;
           break;
       end
       gnStepSize = gnBacktrackFactor * gnStepSize;
   end

   if ~gnAccepted
       error('GaussNewton:LineSearchFailed', ...
           'Gauss-Newton method could not find a decreasing step.');
   end

   gnPoint = gnTrialPoint;
   gnIteration = gnIteration + 1;
   gnPath(:, gnIteration + 1) = gnPoint;
end

gnGradient = ops.rosenbrockGradient(gnPoint(1), gnPoint(2));
gnGradientNorm = norm(gnGradient);
gnValue = ops.rosenbrockFunction(gnPoint(1), gnPoint(2));
gnConverged = gnConverged || gnGradientNorm <= gnTolerance;
gnPath = gnPath(:, 1:gnIteration + 1);

if ~gnConverged
   warning('GaussNewton:NoConvergence', ...
       'Gauss-Newton method did not converge within %d iterations.', gnMaxIterations);
end
fprintf(['Gauss-Newton method: x = %.8g, y = %.8g, f = %.8g, ', ...
   'gradient norm = %.3g, iterations = %d, converged = %d\n'], ...
   gnPoint(1), gnPoint(2), gnValue, gnGradientNorm, gnIteration, gnConverged);

%% Task 3