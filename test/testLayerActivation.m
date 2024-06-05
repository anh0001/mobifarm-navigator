% Add project directories to path
addpath(genpath('src'));   % Add source code directory
addpath(genpath('lib'));   % Add library directory
addpath(genpath('tools')); % Add tools directory
addpath(genpath('models')); % Add models directory

% Load the network
load('model_1fusion_lr1e-02_bs08_data_cave_house_city.mat', 'model'); % Load the model

% Display model input names to verify correctness
disp('Model Input Names:');
disp(model.InputNames);

% Display model layer names to verify correctness
disp('Model Layer Names:');
layerNames = {model.Layers.Name};
disp(layerNames);

% Display model layer connections
disp('Layer Connections:');
disp(model.Connections);

%% Data Preparation
% Load and preprocess data
disp('Loading and preprocessing data...');
[dataTrain, dataVal] = dataPrep(); % Load and preprocess training and validation data

% Reset the datastore to the beginning
reset(dataVal);
stopFlag = false; % Initialize stopFlag
while hasdata(dataVal) && ~stopFlag
    data = read(dataVal);
    img = data{1};    % RGB image
    lidar = data{2};  % Distance map (Lidar)
    pcd = data{3};    % Point cloud data
    groundTruth = data{4}; % Ground truth labels
    
    % Convert inputs to required formats
    img = single(img);
    lidar = single(lidar);
    pcd = single(pcd);
    
    % Display input sizes and check data
    disp('Raw Input Sizes:');
    disp(size(img));
    disp(size(lidar));
    disp(size(pcd));
    disp('Raw Input Data:');
    disp(img(1:5, 1:5, 1)); % Display a small part of the image data
    disp(lidar(1:5, 1:5, 1)); % Display a small part of the lidar data
    disp(pcd(1:5, :)); % Display a small part of the point cloud data
    
    % Convert to dlarray
    img = dlarray(img, 'SSCB');
    lidar = dlarray(lidar, 'SSCB');
    pcd = dlarray(pcd, 'SCB'); % Correcting to 'CB' instead of 'SCB'
    
    % Display dlarray input sizes
    disp('dlarray Input Sizes:');
    disp(size(img));
    disp(size(lidar));
    disp(size(pcd));

    % Test the first layer's activations
    layerName = "conv1"; % Specify the first layer name
    
    % Check if the layer name exists
    if any(strcmp(layerNames, layerName))
        disp(['Layer ', layerName, ' exists. Proceeding to forward pass...']);
        
        % Forward pass to get activations for the first layer
        [~, act] = forward(model, img, lidar, pcd, 'Outputs', layerName);
        
        % Display the size of the activations
        disp('First Layer Activation Size:');
        disp(size(act)); % Example output: [240, 320, 64, 1] for 'conv1' layer
        
        if isempty(act)
            disp('First layer activations are empty.');
        else
            % Visualize the first layer activations
            act = extractdata(act); % Extract data from dlarray
            act = mat2gray(act); % Normalize the activations
            act = imtile(act);   % Tile the activations for visualization
            figure;
            imshow(act);         % Display the activations
        end
    else
        disp(['Layer ', layerName, ' does not exist in the network.']);
    end

    % Wait for a key press
    key = waitforbuttonpress;
    if key == 1 % If the input was a key press
        keyData = get(gcf, 'CurrentKey'); % Get the key that was pressed
        if strcmp(keyData, 'return') % If the key was 'Enter'
            stopFlag = true; % Set stopFlag to true to stop the loop
        end
    end
end

% Remove paths added at the beginning
rmpath(genpath('src'));   % Remove source code directory
rmpath(genpath('lib'));   % Remove library directory
rmpath(genpath('tools')); % Remove tools directory
rmpath(genpath('models')); % Remove models directory
