%% Task 1
clear all
clc

x = linspace(-2, 2, 201);
y = linspace(-2, 2, 201);
[X, Y] = meshgrid(x, y);
F = thefunctions(X, Y);

rgbAnchors = [0 0 1; 0 1 1; 0 1 0; 1 1 0; 1 0 0];
rgbScale = interp1(linspace(0, 1, size(rgbAnchors, 1)), rgbAnchors, ...
    linspace(0, 1, 256));

figure;
meshIndices = 1:5:numel(x);
hSurface = surf(X(meshIndices, meshIndices), Y(meshIndices, meshIndices), ...
    F(meshIndices, meshIndices), 'EdgeColor', [0.2 0.2 0.2], ...
    'LineWidth', 0.25, 'DisplayName', 'f(x,y)');
hold on;
hMinimum = scatter3(-1, 1, 0, 90, [1 0 1], 'filled', ...
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
hMinimum2D = plot(-1, 1, 'o', 'MarkerSize', 14 / sqrt(2), ...
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

%% Task 3