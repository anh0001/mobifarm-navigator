function [trainedModel, trainInfo] = trainModel(model, dataTrain, dataVal)
    % trainModel.m
    % This function configures and conducts the training of the deep learning
    % model for mobile robot navigation using multiple sensors' data.
    % It returns the trained model and training information.

    disp('Configuring training options...');
    
    %% Configure Training Options
    options = trainingOptions('sgdm', ...
        'InitialLearnRate', 1e-3, ...  % Start with a higher learning rate
        'Momentum', 0.9, ...  % Typical value for momentum
        'LearnRateSchedule', 'piecewise', ...  % Implement piecewise learning rate schedule
        'LearnRateDropPeriod', 10, ...  % Drop learning rate every 10 epochs
        'LearnRateDropFactor', 0.1, ...  % Drop learning rate by a factor of 0.1
        'MaxEpochs', 30, ...
        'MiniBatchSize', 8, ...
        'ValidationData', dataVal, ...
        'ValidationFrequency', 60, ...
        'Verbose', true, ...
        'Plots', 'training-progress', ...
        'ExecutionEnvironment', 'auto', ...
        'CheckpointPath', 'data/checkpoints');  % custom function

    %% Train the Model
    disp('Starting model training...');
    [trainedModel, trainInfo] = trainnet(dataTrain, model, 'mse', options);

    disp('Model training complete.');
end

function stop = stopIfAccuracyNotImproving(info, N)
    stop = false;
    if strcmp(info.State, 'done')
        return;
    end
    % Ensure ValidationLoss is a vector and has enough entries
    if isfield(info, 'ValidationLoss') && numel(info.ValidationLoss) >= N
        % Check if the most recent ValidationLoss values have improved
        if info.Epoch > N && any(info.ValidationLoss(end-N:end) > min(info.ValidationLoss(1:end-N)))
            stop = true;
            disp('Stopping early due to no improvement in validation loss.');
        end
    end
end