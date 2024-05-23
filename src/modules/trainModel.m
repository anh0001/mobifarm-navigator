function [trainedModel, trainInfo] = trainModel(model, dataTrain, dataVal)
    % trainModel.m
    % This function configures and conducts the training of the deep learning
    % model for mobile robot navigation using multiple sensors' data.
    % It returns the trained model and training information.

    disp('Configuring training options...');
    
    %% Configure Training Options
    options = trainingOptions('adam', ...
        'InitialLearnRate', 1e-4, ...
        'MaxEpochs', 30, ...
        'MiniBatchSize', 32, ...
        'Shuffle', 'every-epoch', ...
        'ValidationData', dataVal, ...
        'ValidationFrequency', 30, ...
        'Verbose', true, ...
        'Plots', 'training-progress', ...
        'ExecutionEnvironment', 'auto', ...
        'CheckpointPath', 'data/checkpoints', ...
        'OutputFcn', @(info)stopIfAccuracyNotImproving(info,20));  % custom function

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