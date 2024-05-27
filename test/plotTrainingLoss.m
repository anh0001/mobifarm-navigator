% Load trainInfo.mat

% Plot the training and validation loss
figure,
plot(trainInfo.TrainingHistory.Loss, 'b-', 'LineWidth', 2);
hold on;
plot(trainInfo.ValidationHistory.Loss, 'r--', 'LineWidth', 2);
hold off;

% Set x-axis range
xlim([800, 1500]);
% ylim([0, 1]);

% Add labels and title
xlabel('Iteration');
ylabel('Loss');
title('Training and Validation Loss');

% Add legend
legend('Training Loss', 'Validation Loss');