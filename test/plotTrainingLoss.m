% Existing code
close all
% load('trainInfo.mat')

% Interpolate the validation loss to match the training iterations
trainingIterations = length(trainInfo.TrainingHistory.Loss);
validationIterations = length(trainInfo.ValidationHistory.Loss);
validationLossInterpolated = interp1(linspace(1, trainingIterations, validationIterations), trainInfo.ValidationHistory.Loss, 1:trainingIterations, 'linear');

% Plot the training and interpolated validation loss
figure,
plot(trainInfo.TrainingHistory.Loss, 'b-', 'LineWidth', 2);
hold on;
plot(validationLossInterpolated, 'r-', 'LineWidth', 2);
hold off;

% Set x-axis range
xlim([0, 250]);
% ylim([0, 1]);

grid on;

% Add labels and title
xlabel('Iteration', 'FontSize', 14);
ylabel('Loss', 'FontSize', 14);

% Add legend
legend('Training Loss', 'Validation Loss', 'FontSize', 14);

% New code to increase font size of x and y axis tick values
ax = gca; % get current axes
ax.XAxis.FontSize = 12; % set x-axis font size
ax.YAxis.FontSize = 12; % set y-axis font size
