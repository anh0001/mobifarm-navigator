function [dataTrain, dataVal] = dataPrep()
    % dataPrep.m
    % This function loads and preprocesses data from the 'data/raw' directory
    % and organizes it into training and validation datasets stored in 'data/processed'.
    
    disp('Starting data preparation...');
    
    %% Directories
    processedDir = 'data/processed/'; % Directory where processed data is stored
    
    %% Dataset Folders
    datasets = {'data_cave'}; % List of datasets to be processed
    
    %% Initialize cell arrays to collect datastores
    trainDatastores = {}; % Cell array to hold training datastores
    valDatastores = {};   % Cell array to hold validation datastores
    
    %% Load and Preprocess Data
    for i = 1:length(datasets)
        dataset = datasets{i}; % Current dataset
        
        % Load and Preprocess Training Data
        trainDir = fullfile(processedDir, dataset, 'train'); % Path to training data
        trainDatastores{end+1} = preprocessData(trainDir); % Preprocess and store training data
        
        % Load and Preprocess Validation Data
        valDir = fullfile(processedDir, dataset, 'val'); % Path to validation data
        valDatastores{end+1} = preprocessData(valDir); % Preprocess and store validation data
    end
    
    %% Combine all the datastores
    dataTrain = combine(trainDatastores{:}); % Combine all training datastores into one
    dataVal = combine(valDatastores{:});     % Combine all validation datastores into one
    
    disp('Data preparation completed.');
end

function combineDS = preprocessData(folderPath)
    % preprocessData
    % This function processes data from a specific folder, performing operations
    % such as loading images, labels, point cloud data, and maps.
    
    disp(['Processing data in: ', folderPath]);
    
    % Find all relevant files recursively
    imgFiles = findFilesRecursively(folderPath, '**/cam_front/*.jpg'); % Find image files
    labelFiles = findFilesRecursively(folderPath, '**/labels/*.txt'); % Find label files
    lidarFiles = findFilesRecursively(folderPath, '**/realtime_map/*.jpg'); % Find lidar map files
    pcdFiles = findFilesRecursively(folderPath, '**/pcd_preprocessed/*.npy'); % Find point cloud data files
    
    % Extract full file paths
    imgFiles = fullfile({imgFiles.folder}, {imgFiles.name}); % Full paths to image files
    labelFiles = fullfile({labelFiles.folder}, {labelFiles.name}); % Full paths to label files
    lidarFiles = fullfile({lidarFiles.folder}, {lidarFiles.name}); % Full paths to lidar files
    pcdFiles = fullfile({pcdFiles.folder}, {pcdFiles.name}); % Full paths to point cloud data files
    
    % Load Camera Images using imageDatastore
    imgDS = imageDatastore(imgFiles, 'FileExtensions', {'.jpg'}, 'ReadFcn', @readImage); % Image datastore for camera images
    
    % Load Real-time Maps using imageDatastore
    lidarDS = imageDatastore(lidarFiles, 'FileExtensions', {'.jpg'}, 'ReadFcn', @readImage); % Image datastore for lidar maps
    
    % Load Processed Point Cloud Data using fileDatastore
    pcdDS = fileDatastore(pcdFiles, 'ReadFcn', @bacaNPY, 'FileExtensions', {'.npy'}); % File datastore for point cloud data
    
    % Load Labels into an array
    labelData = cellfun(@readmatrix, labelFiles, 'UniformOutput', false); % Load labels from text files
    labelDataMatrix = vertcat(labelData{:}); % Convert cell array to a matrix
    
    % Create an arrayDatastore for linear and angular velocities
    labelsDS = arrayDatastore(labelDataMatrix, 'IterationDimension', 1); % Array datastore for labels
    
    % Combine the datastores into a single datastore
    combineDS = combine(imgDS, lidarDS, pcdDS, labelsDS); % Combine all datastores into one
end

function files = findFilesRecursively(folderPath, pattern)
    % findFilesRecursively
    % This function finds files matching a given pattern recursively within a folder.
    files = dir(fullfile(folderPath, pattern)); % Find files matching the pattern
    files = files(~[files.isdir]); % Filter out directories
end

function data = readImage(filename)
    % readImage
    % Custom read function for imageDatastore to read images.
    data = imread(filename); % Read the image file
end

function data = bacaNPY(filename)
    % bacaNPY
    % Custom read function for fileDatastore to load .npy files.
    data = readNPY(filename); % Read the .npy file (Assuming you have a readNPY function available)
    
    % Check if data is empty
    if isempty(data)
        error('Data is empty. Please ensure the fileDatastore is correctly loaded with data.');
    end
end
