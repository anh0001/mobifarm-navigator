function [trainedModel, trainInfo] = trainModel(model, dataTrain, dataVal)
    % trainModel.m
    % This function configures and conducts the training of the deep learning
    % model for mobile robot navigation using multiple sensors' data.
    % It returns the trained model and training information.

    disp('Configuring training options...');
    
    %% Configure Training Options
    numEpochs = 30;
    miniBatchSize = 32;
    learnRate = 1e-4;
    validationFrequency = 30;
    
    %% Initialize Training Variables
    trainLoss = [];
    valLoss = [];
    iteration = 0;

    %% Custom Training Loop
    disp('Starting model training...');
    for epoch = 1:numEpochs
        % Shuffle the data
        dataTrain = shuffle(dataTrain);
        
        % Reset the data source
        reset(dataTrain);
        
        while hasdata(dataTrain)
            iteration = iteration + 1;
            
            % Read mini-batch of data
            batchData = read(dataTrain);
            X = batchData(:, 1:3); % Inputs: images, LiDAR, point clouds
            T = batchData(:, 4:5); % Targets: linear and angular velocities

            % Convert mini-batch data to dlarray
            dlX = cellfun(@(x) dlarray(x, 'SSCB'), X, 'UniformOutput', false);
            dlT = dlarray(T, 'CB');
            
            % If GPU is available, then convert data to gpuArray
            if canUseGPU
                dlX = cellfun(@gpuArray, dlX, 'UniformOutput', false);
                dlT = gpuArray(dlT);
            end
            
            % Evaluate the model gradients and loss using dlfeval and the modelGradients function
            [gradients, loss] = dlfeval(@modelGradients, model, dlX, dlT);
            
            % Update the network parameters
            model = adamupdate(model, gradients, learnRate);
            
            % Record training loss
            trainLoss = [trainLoss; loss];

            % Validate the model
            if mod(iteration, validationFrequency) == 0
                valLoss = [valLoss; computeValidationLoss(model, dataVal)];
            end
            
            % Display progress
            disp(['Epoch ' num2str(epoch) ', Iteration ' num2str(iteration) ', Loss ' num2str(loss)]);
        end
    end

    disp('Model training complete.');

    trainedModel = model;
    trainInfo = struct('TrainLoss', trainLoss, 'ValLoss', valLoss);

    %% Helper Functions
    function [gradients, loss] = modelGradients(model, X, T)
        % Forward pass
        Y = forward(model, X{:});
        
        % Compute mean squared error loss
        loss = mse(Y, T);
        
        % Compute gradients
        gradients = dlgradient(loss, model.Learnables);
    end

    function valLoss = computeValidationLoss(model, dataVal)
        % Initialize validation loss
        valLoss = 0;
        numValData = 0;
        
        % Reset the data source
        reset(dataVal);
        
        while hasdata(dataVal)
            % Read mini-batch of data
            batchData = read(dataVal);
            X = batchData(:, 1:3); % Inputs: images, LiDAR, point clouds
            T = batchData(:, 4:5); % Targets: linear and angular velocities

            % Convert mini-batch data to dlarray
            dlX = cellfun(@(x) dlarray(x, 'SSCB'), X, 'UniformOutput', false);
            dlT = dlarray(T, 'CB');
            
            % If GPU is available, then convert data to gpuArray
            if canUseGPU
                dlX = cellfun(@gpuArray, dlX, 'UniformOutput', false);
                dlT = gpuArray(dlT);
            end
            
            % Forward pass
            Y = forward(model, dlX{:});
            
            % Compute mean squared error loss
            loss = mse(Y, dlT);
            
            % Accumulate validation loss
            valLoss = valLoss + loss;
            numValData = numValData + 1;
        end
        
        % Average validation loss
        valLoss = valLoss / numValData;
    end
end
