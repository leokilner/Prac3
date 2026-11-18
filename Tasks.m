%% Creating function and plotting it with the minimum point and the contour plot of the function
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
hMinimum = scatter3(1, 1, 0, 45, [1 0 1], 'filled', ...
    'MarkerEdgeColor', 'k', 'DisplayName', 'Minimum (1, 1, 0)');
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
hMinimum2D = plot(1, 1, 'o', 'MarkerSize', 7 / sqrt(2), ...
    'MarkerFaceColor', [1 0 1], 'MarkerEdgeColor', 'k', ...
    'DisplayName', 'Minimum (1, 1)');
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

%% Implementing gradient descent, Newton's method and Gauss-Newton method; using the same 3 random starting points for each method and comparing the results

for i = 1:3 % Implementing a for loop to run the optimization methods for 3 random starting points

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

    % Plotting all three optimization paths on the 2D contour plot
    figure;
    contour(X, Y, F, logspace(-3, 3.5, 35), ...
        'LineColor', [0.4 0.4 0.4], 'LineWidth', 0.75);
    hold on;
    hGradientDescent = plot(gdPath(1, :), gdPath(2, :), '-o', ...
        'Color', [0.85 0.2 0.2], 'MarkerFaceColor', [0.85 0.2 0.2], ...
        'MarkerSize', 4, 'LineWidth', 1.2);
    hNewton = plot(newtonPath(1, :), newtonPath(2, :), '-o', ...
        'Color', [0.1 0.35 0.85], 'MarkerFaceColor', [0.1 0.35 0.85], ...
        'MarkerSize', 4, 'LineWidth', 1.2);
    hGaussNewton = plot(gnPath(1, :), gnPath(2, :), '-o', ...
        'Color', [0.1 0.6 0.25], 'MarkerFaceColor', [0.1 0.6 0.25], ...
        'MarkerSize', 4, 'LineWidth', 1.2);
    hInitialPoint = plot(x0, y0, 's', 'MarkerSize', 9, ...
        'MarkerFaceColor', [1 0.85 0], 'MarkerEdgeColor', 'k');
    hTrueMinimum = plot(1, 1, 'o', 'MarkerSize', 7 / sqrt(2), ...
        'MarkerFaceColor', [1 0 1], 'MarkerEdgeColor', 'k');
    hold off;
    xlabel('x');
    ylabel('y');
    title(sprintf('Optimization Paths from Initial Point (%.4f, %.4f)', x0, y0));
    legend([hGradientDescent, hNewton, hGaussNewton, hInitialPoint, hTrueMinimum], ...
        {sprintf('Gradient descent (%d iterations)', gdIteration), ...
        sprintf('Newton method (%d iterations)', newtonIteration), ...
        sprintf('Gauss-Newton method (%d iterations)', gnIteration), ...
        'Initial point', 'True minimum (1, 1)'}, ...
        'Location', 'best');
    axis tight;
    grid on;

end % End of the for loop for converging with the first three methods, from 3 random starting points

%% Implementing Nelder-Mead Simplex algorithm from 3 distinct random initialisations and comparing the results
nmReflection = 1;
nmExpansion = 2;
nmContraction = 0.5;
nmShrink = 0.5;
nmFunctionTolerance = 1e-10;
nmSimplexTolerance = 1e-6;
nmMaxIterations = 2000;
nmAreaTolerance = 1e-6;
nmFigure = [];
nmFinished = false;

for nmInitialization = 1:3
    if nmFinished
        break;
    end

    while true
        nmSimplex = [x(1) + rand(1, 3) * (x(end) - x(1)); ...
            y(1) + rand(1, 3) * (y(end) - y(1))];
        nmArea = abs(det([nmSimplex(:, 2) - nmSimplex(:, 1), ...
            nmSimplex(:, 3) - nmSimplex(:, 1)])) / 2;
        if nmArea > nmAreaTolerance
            break;
        end
    end

    nmHistory = cell(nmMaxIterations + 1, 1);
    nmHistory{1} = nmSimplex;
    nmIteration = 0;
    nmConverged = false;

    while nmIteration < nmMaxIterations
        nmValues = zeros(1, 3);
        for nmVertex = 1:3
            nmValues(nmVertex) = ops.rosenbrockFunction( ...
                nmSimplex(1, nmVertex), nmSimplex(2, nmVertex));
        end
        [nmValues, nmOrder] = sort(nmValues);
        nmSimplex = nmSimplex(:, nmOrder);

        nmDistances = sqrt(sum((nmSimplex(:, 2:3) - nmSimplex(:, 1)).^2, 1));
        if max(nmDistances) <= nmSimplexTolerance && ...
                max(abs(nmValues - nmValues(1))) <= nmFunctionTolerance
            nmConverged = true;
            break;
        end

        nmCentroid = mean(nmSimplex(:, 1:2), 2);
        nmWorst = nmSimplex(:, 3);
        nmReflected = nmCentroid + nmReflection * (nmCentroid - nmWorst);
        nmReflectedValue = ops.rosenbrockFunction(nmReflected(1), nmReflected(2));

        if nmReflectedValue < nmValues(1)
            nmExpanded = nmCentroid + nmExpansion * (nmReflected - nmCentroid);
            nmExpandedValue = ops.rosenbrockFunction(nmExpanded(1), nmExpanded(2));
            if nmExpandedValue < nmReflectedValue
                nmSimplex(:, 3) = nmExpanded;
            else
                nmSimplex(:, 3) = nmReflected;
            end
        elseif nmReflectedValue < nmValues(2)
            nmSimplex(:, 3) = nmReflected;
        elseif nmReflectedValue < nmValues(3)
            nmOutsideContracted = nmCentroid + ...
                nmContraction * (nmReflected - nmCentroid);
            nmOutsideValue = ops.rosenbrockFunction( ...
                nmOutsideContracted(1), nmOutsideContracted(2));
            if nmOutsideValue <= nmReflectedValue
                nmSimplex(:, 3) = nmOutsideContracted;
            else
                for nmShrinkVertex = 2:3
                    nmSimplex(:, nmShrinkVertex) = nmSimplex(:, 1) + ...
                        nmShrink * (nmSimplex(:, nmShrinkVertex) - nmSimplex(:, 1));
                end
            end
        else
            nmInsideContracted = nmCentroid + ...
                nmContraction * (nmWorst - nmCentroid);
            nmInsideValue = ops.rosenbrockFunction( ...
                nmInsideContracted(1), nmInsideContracted(2));
            if nmInsideValue < nmValues(3)
                nmSimplex(:, 3) = nmInsideContracted;
            else
                for nmShrinkVertex = 2:3
                    nmSimplex(:, nmShrinkVertex) = nmSimplex(:, 1) + ...
                        nmShrink * (nmSimplex(:, nmShrinkVertex) - nmSimplex(:, 1));
                end
            end
        end

        nmIteration = nmIteration + 1;
        nmHistory{nmIteration + 1} = nmSimplex;
    end

    nmFinalValues = zeros(1, 3);
    for nmVertex = 1:3
        nmFinalValues(nmVertex) = ops.rosenbrockFunction( ...
            nmSimplex(1, nmVertex), nmSimplex(2, nmVertex));
    end
    [nmFinalValues, nmFinalOrder] = sort(nmFinalValues);
    nmSimplex = nmSimplex(:, nmFinalOrder);
    nmFinalDistances = sqrt(sum((nmSimplex(:, 2:3) - nmSimplex(:, 1)).^2, 1));
    nmConverged = nmConverged || ...
        (max(nmFinalDistances) <= nmSimplexTolerance && ...
        max(abs(nmFinalValues - nmFinalValues(1))) <= nmFunctionTolerance);
    nmHistory{nmIteration + 1} = nmSimplex;

    nmHistory = nmHistory(1:nmIteration + 1);
    if ~nmConverged
        warning('NelderMead:NoConvergence', ...
            'Initialization %d reached the iteration limit without convergence.', ...
            nmInitialization);
    end

    if isempty(nmFigure)
        nmFigure = figure('Name', 'Nelder-Mead Simplex Animation');
    else
        clf(nmFigure);
    end
    nmAxes = axes('Parent', nmFigure);
    contour(nmAxes, X, Y, F, logspace(-3, 3.5, 35), ...
        'LineColor', [0.4 0.4 0.4], 'LineWidth', 0.75);
    hold(nmAxes, 'on');
    nmSimplexLine = plot(nmAxes, NaN, NaN, '-o', ...
        'Color', [0.85 0.2 0.2], 'MarkerFaceColor', [0.85 0.2 0.2], ...
        'MarkerSize', 5, 'LineWidth', 1.5, 'DisplayName', 'Current simplex');
    nmTrueMinimum = plot(nmAxes, 1, 1, 'o', 'MarkerSize', 7 / sqrt(2), ...
        'MarkerFaceColor', [1 0 1], 'MarkerEdgeColor', 'k', ...
        'DisplayName', 'True minimum (1, 1)');
    hold(nmAxes, 'off');
    xlim(nmAxes, [x(1) x(end)]);
    ylim(nmAxes, [y(1) y(end)]);
    xlabel(nmAxes, 'x');
    ylabel(nmAxes, 'y');
    if nmConverged
        title(nmAxes, sprintf('Nelder-Mead: Initialization %d of 3', nmInitialization));
    else
        title(nmAxes, sprintf( ...
            'Nelder-Mead: Initialization %d of 3 (not converged)', nmInitialization));
    end
    grid(nmAxes, 'on');
    legend(nmAxes, [nmSimplexLine, nmTrueMinimum], ...
        {'Current simplex', 'True minimum (1, 1)'}, 'Location', 'northeast');
    nmIterationCounter = text(nmAxes, 0.02, 0.98, '', ...
        'Units', 'normalized', 'VerticalAlignment', 'top', ...
        'BackgroundColor', 'w', 'Margin', 3);

    nmReplay = true;
    while nmReplay && ~nmFinished
        for nmFrame = 1:numel(nmHistory)
            if ~isgraphics(nmFigure)
                nmFinished = true;
                break;
            end
            nmFrameSimplex = nmHistory{nmFrame};
            nmClosedSimplex = [nmFrameSimplex, nmFrameSimplex(:, 1)];
            set(nmSimplexLine, 'XData', nmClosedSimplex(1, :), ...
                'YData', nmClosedSimplex(2, :));
            set(nmIterationCounter, 'String', ...
                sprintf('Iteration %d / %d', nmFrame - 1, nmIteration));
            drawnow;
            pause(0.2);
        end

        if nmFinished
            break;
        end

        if nmInitialization < 3
            nmChoice = questdlg( ...
                sprintf('Initialization %d is complete.', nmInitialization), ...
                'Nelder-Mead animation', 'Next initialisation', ...
                'Show again', 'Next initialisation');
            if ~isgraphics(nmFigure)
                nmFinished = true;
                nmReplay = false;
            elseif strcmp(nmChoice, 'Show again')
                nmReplay = true;
            else
                nmReplay = false;
            end
        else
            nmChoice = questdlg( ...
                'The third initialization is complete.', ...
                'Nelder-Mead animation', 'Show again', 'Finish', 'Finish');
            if ~isgraphics(nmFigure)
                nmFinished = true;
                nmReplay = false;
            elseif strcmp(nmChoice, 'Show again')
                nmReplay = true;
            else
                close(nmFigure);
                nmFinished = true;
                nmReplay = false;
            end
        end
    end
end
