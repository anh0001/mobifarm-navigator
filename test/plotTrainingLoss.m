% Existing code
close all
% Load trainInfo.mat

% Plot the training and validation loss
figure,
plot(trainInfo.TrainingHistory.Loss, 'b-', 'LineWidth', 2);
hold on;
plot(trainInfo.ValidationHistory.Loss, 'r--', 'LineWidth', 2);
hold off;

% Set x-axis range
xlim([20, 150]);
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