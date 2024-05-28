function evaluateModel(dataTest, model)
    % Debugging: Display the number of observations in each underlying datastore
    numObservations = arrayfun(@(x) readall(x), dataTest.UnderlyingDatastores, 'UniformOutput', false);
    numObservations = cellfun(@(x) size(x, 1), numObservations);
    disp('Number of observations in each underlying datastore:');
    disp(numObservations);
    
    % Calculate total number of observations manually
    totalObservations = sum(numObservations);
    disp('Total number of observations in combined datastore:');
    disp(totalObservations);
    
    try
        % Attempt to subset the combined datastore
        batchDataTest = subset(dataTest, totalObservations);
    catch ME
        disp('Error with subset operation:');
        disp(ME.message);
        return;
    end
    
    while hasdata(batchDataTest)
        batch = read(batchDataTest);
        inputs = cell(1, numel(batch.UnderlyingDatastores) - 1);
        
        for i = 1:numel(inputs)
            data = readall(batch.UnderlyingDatastores{i});
            
            if iscell(data)
                if i == 1 || i == 2
                    data = cat(4, data{:});
                elseif i == 3
                    data = cat(3, data{:});
                end
            end
            
            if ~isa(data, 'single') && ~isa(data, 'double') && ~isa(data, 'logical')
                data = single(data);
            end
            
            if i == 1 || i == 2
                if ~isa(data, 'dlarray')
                    data = dlarray(data, 'SSCB');
                end
            elseif i == 3
                if ~isa(data, 'dlarray')
                    data = dlarray(data, 'SCB');
                end
            end
            
            inputs{i} = data;
        end
        
        predictedOutputs = predict(model, inputs{:});
        
        labelsDS = batch.UnderlyingDatastores{end};
        actualOutputs = readall(labelsDS);
        if iscell(actualOutputs)
            actualOutputs = cell2mat(actualOutputs);
        end
        
        predictedOutputs = extractdata(predictedOutputs);
        predictedOutputs = permute(predictedOutputs, [2, 1]);
        
        error = predictedOutputs - actualOutputs;
        squaredError = sum(error .^ 2, 'all');
        
        batchError.squaredError = squaredError;
        batchError.total = size(actualOutputs, 1);
    end
end



% function rmse = evaluateModel(model, dataTest, batchSize)
% % evaluateModel
% % This function evaluates the model in batches to handle large datasets.
% % Inputs:
% %   - model: the trained model
% %   - dataTest: the test dataset
% %   - batchSize: size of each batch
% % Output:
% %   - rmse: overall Root Mean Squared Error of the model

% numObservations = numel(dataTest.UnderlyingDatastores{1}.Files);
% totalSquaredError = 0;
% totalPredictions = 0;

% for startIdx = 1:batchSize:numObservations
%     endIdx = min(startIdx + batchSize - 1, numObservations);
%     batchIndices = startIdx:endIdx;
    
%     try
%         % Attempt to subset the combined datastore
%         batchDataTest = subset(dataTest, batchIndices);
%     catch ME
%         disp('Error with subset operation:');
%         disp(ME.message);
%         disp('Available indices:');
%         disp(dataTest.NumObservations);
%         return;
%     end
    
%     % Evaluate the batch
%     batchError = evaluateBatch(model, batchDataTest);
    
%     % Update overall squared error and prediction count
%     totalSquaredError = totalSquaredError + batchError.squaredError;
%     totalPredictions = totalPredictions + batchError.total;
% end

% % Compute the overall RMSE
% mse = totalSquaredError / totalPredictions;
% rmse = sqrt(mse);
% end

% function batchError = evaluateBatch(model, batchDataTest)
% % evaluateBatch
% % This function evaluates a batch of data.
% % Inputs:
% %   - model: the trained model
% %   - batchDataTest: a batch of test data
% % Output:
% %   - batchError: a structure with squared error and total predictions

% % Check the input names of the model
% inputNames = model.InputNames;

% % Prepare the inputs for the predict function
% inputs = cell(1, numel(inputNames));
% for i = 1:numel(inputNames)
%     % Extract the i-th input from the combined datastore
%     data = readall(batchDataTest.UnderlyingDatastores{i});
    
%     % If data is a cell array, extract its contents
%     if iscell(data)
%         if i == 1 || i == 2
%             % First and second inputs are images in SSC format
%             data = cat(4, data{:}); % Concatenate along the 4th dimension to form SxSxCxB
%         elseif i == 3
%             % Third input is in SC format
%             data = cat(3, data{:}); % Concatenate along the 3rd dimension to form SxCxB
%         end
%     end
    
%     % Convert to single if necessary
%     if ~isa(data, 'single') && ~isa(data, 'double') && ~isa(data, 'logical')
%         data = single(data);
%     end
    
%     % Convert to dlarray if necessary
%     if i == 1 || i == 2
%         % First and second inputs
%         if ~isa(data, 'dlarray')
%             data = dlarray(data, 'SSCB'); % Spatial, Spatial, Channel, Batch
%         end
%     elseif i == 3
%         % Third input
%         if ~isa(data, 'dlarray')
%             data = dlarray(data, 'SCB'); % Spatial, Channel, Batch
%         end
%     end
    
%     inputs{i} = data;
% end

% % Predict the outputs using the model
% predictedOutputs = predict(model, inputs{:});

% % Extract the ground truth labels from the batch data
% labelsDS = batchDataTest.UnderlyingDatastores{end}; % Assuming the labels are the last datastore
% actualOutputs = readall(labelsDS);
% % If data is a cell array, extract its contents
% if iscell(actualOutputs)
%     actualOutputs = cell2mat(actualOutputs);
% end

% % Calculate squared error
% % Convert predictedOutputs to a double array
% predictedOutputs = extractdata(predictedOutputs);
% % Ensure predictedOutputs and actualOutputs have compatible dimensions
% predictedOutputs = permute(predictedOutputs, [2, 1]);
% % Compute the error
% error = predictedOutputs - actualOutputs;
% % Compute the squared error
% squaredError = sum(error .^ 2, 'all');

% % Store results in a structure
% batchError.squaredError = squaredError;
% batchError.total = size(actualOutputs, 1); % Number of data points in the batch
% end
