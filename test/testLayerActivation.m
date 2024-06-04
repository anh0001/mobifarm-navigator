% Add project directories to path
addpath(genpath('src'));   % Add source code directory
addpath(genpath('lib'));   % Add library directory
addpath(genpath('tools')); % Add tools directory
addpath(genpath('models')); % Add tools directory

% Load the network
load('model_1fusion_lr1e-02_bs08_data_cave_house_city.mat', 'model'); % Load the model

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
    
    % Convert to dlarray
    img = dlarray(img, 'SSCB');
    lidar = dlarray(lidar, 'SSCB');
    pcd = dlarray(pcd, 'SCB');
    
    % Get the output of the specified layer
    layerName = "img_last_layer"; % Specify the layer name
    act = activations(model, {img, lidar, pcd}, layerName);
    
    % Display the size of the activations
    disp(size(act)); % Example output: [55, 55, 16]
    
    % Visualize the activations
    act = mat2gray(act); % Normalize the activations
    act = imtile(act);   % Tile the activations for visualization
    figure;
    imshow(act);         % Display the activations

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
rmpath(genpath('models')); % Remove tools directory