function [trainedModel, trainInfo] = trainModel(model, dataTrain, dataVal)
    % trainModel.m
    % This function configures and conducts the training of the deep learning
    % model for mobile robot navigation using multiple sensors' data.
    % It returns the trained model and training information.

    disp('Configuring training options...');
    
    options = trainingOptions('sgdm', ...
        'InitialLearnRate', 1e-2, ...  % Initial learning rate
        'Momentum', 0.9, ...  % Momentum value to accelerate gradients vectors
        'LearnRateSchedule', 'piecewise', ...  % Schedule for changing learning rate
        'LearnRateDropPeriod', 50, ...  % Number of epochs after which learning rate drops
        'LearnRateDropFactor', 0.1, ...  % Factor by which learning rate decreases
        'MaxEpochs', 150, ...  % Maximum number of training epochs
        'MiniBatchSize', 8, ...  % Number of samples per gradient update
        'ValidationData', dataVal, ...  % Validation data for monitoring training progress
        'ValidationFrequency', 50, ...  % Frequency of validation
        'Verbose', true, ...  % Display training progress information
        'Plots', 'training-progress', ...  % Plot training progress
        'ExecutionEnvironment', 'auto', ...  % Automatically select execution environment
        'CheckpointPath', 'data/checkpoints');  % Path to save checkpoint data

    %% Train the Model
    disp('Starting model training...');
    [trainedModel, trainInfo] = trainnet(dataTrain, model, 'mse', options); % Train the model with mean squared error loss

    disp('Model training complete.');
end

function stop = stopIfAccuracyNotImproving(info, N)
    % stopIfAccuracyNotImproving
    % This function checks if the validation loss has stopped improving for N epochs.
    % If so, it stops the training early to prevent overfitting.
    % Returns a logical value indicating whether to stop training.

    stop = false; % Initialize stop flag as false

    if strcmp(info.State, 'done') % If training is done, return false
        return;
    end

    % Ensure ValidationLoss is a vector and has enough entries
    if isfield(info, 'ValidationLoss') && numel(info.ValidationLoss) >= N
        % Check if the most recent ValidationLoss values have not improved
        if info.Epoch > N && any(info.ValidationLoss(end-N:end) > min(info.ValidationLoss(1:end-N)))
            stop = true; % Set stop flag to true
            disp('Stopping early due to no improvement in validation loss.');
        end
    end
end
