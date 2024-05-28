function [dataTrain, dataVal] = dataPrep()
    % dataPrep.m
    % This function loads and preprocesses data from the 'data/raw' directory
    % and organizes it into training and validation datasets stored in 'data/processed'.
    
    disp('Starting data preparation...');
    
    %% Directories
    processedDir = 'data/processed/'; % Directory where processed data is stored
    
    %% Dataset Folders
    datasets = {'data_cave', 'data_city', 'data_house'}; % List of datasets to be processed
    
    %% Initialize cell arrays to collect datastores
    trainDatastores = {}; % Cell array to hold training datastores
    valDatastores = {};   % Cell array to hold validation datastores
    
    %% Load and Preprocess Data
    for i = 1:length(datasets)
        dataset = datasets{i}; % Current dataset
        
        % Load and Preprocess Training Data
        trainDir = fullfile(processedDir, dataset, 'train'); % Path to training data
        trainDS = preprocessData(trainDir); % Preprocess and store training data
        trainDatastores = [trainDatastores, trainDS]; % Concatenate cell arrays
        
        % Load and Preprocess Validation Data
        valDir = fullfile(processedDir, dataset, 'val'); % Path to validation data
        valDS = preprocessData(valDir); % Preprocess and store validation data
        valDatastores = [valDatastores, valDS]; % Concatenate cell arrays
    end
    
    %% Combine all the datastores
    dataTrain = combine(trainDatastores{:}); % Combine all training datastores into one
    dataVal = combine(valDatastores{:});     % Combine all validation datastores into one
    
    disp('Data preparation completed.');
end

function combinedDatastores = preprocessData(folderPath)
    % preprocessData
    % This function processes data from a specific folder, performing operations
    % such as loading images, labels, point cloud data, and maps.
    
    % Find subdirectories within the given folder path
    subdirs = dir(folderPath);
    subdirs = subdirs([subdirs.isdir] & ~ismember({subdirs.name}, {'.', '..'}));
    
    combinedDatastores = {};
    
    for k = 1:length(subdirs)
        subdirPath = fullfile(folderPath, subdirs(k).name);
        disp(['Processing data in: ', subdirPath]);
        
        % Find all relevant files recursively
        imgFiles = findFilesRecursively(subdirPath, '**/cam_front/*.jpg'); % Find image files
        labelFiles = findFilesRecursively(subdirPath, '**/labels/*.txt'); % Find label files
        lidarFiles = findFilesRecursively(subdirPath, '**/realtime_map/*.jpg'); % Find lidar map files
        pcdFiles = findFilesRecursively(subdirPath, '**/pcd_preprocessed/*.npy'); % Find point cloud data files
        
        % Extract filenames without extensions
        imgNames = extractFileNames(imgFiles);
        labelNames = extractFileNames(labelFiles);
        lidarNames = extractFileNames(lidarFiles);
        pcdNames = extractFileNames(pcdFiles);
        
        % Find common filenames across all categories
        commonNames = intersect(imgNames, intersect(labelNames, intersect(lidarNames, pcdNames)));
        
        % Filter files to only include those with matching filenames
        imgFiles = filterFilesByNames(imgFiles, commonNames);
        labelFiles = filterFilesByNames(labelFiles, commonNames);
        lidarFiles = filterFilesByNames(lidarFiles, commonNames);
        pcdFiles = filterFilesByNames(pcdFiles, commonNames);
        
        % Load Camera Images using imageDatastore
        imgDS = imageDatastore(fullfile({imgFiles.folder}, {imgFiles.name}), 'FileExtensions', {'.jpg'}, 'ReadFcn', @readImage); % Image datastore for camera images
        
        % Load Real-time Maps using imageDatastore
        lidarDS = imageDatastore(fullfile({lidarFiles.folder}, {lidarFiles.name}), 'FileExtensions', {'.jpg'}, 'ReadFcn', @readImage); % Image datastore for lidar maps
        
        % Load Processed Point Cloud Data using fileDatastore
        pcdDS = fileDatastore(fullfile({pcdFiles.folder}, {pcdFiles.name}), 'ReadFcn', @bacaNPY, 'FileExtensions', {'.npy'}); % File datastore for point cloud data

        % Load Labels Data using fileDatastore
        labelsDS = fileDatastore(fullfile({labelFiles.folder}, {labelFiles.name}), 'ReadFcn', @readmatrix, 'FileExtensions', {'.txt'}); % File datastore for labels data
        
        % Combine the datastores into a single datastore
        combinedDS = combine(imgDS, lidarDS, pcdDS, labelsDS); % Combine all datastores into one
        
        combinedDatastores{end+1} = combinedDS; % Store the combined datastore in a cell array
    end
end

function files = findFilesRecursively(folderPath, pattern)
    % findFilesRecursively
    % This function finds files matching a given pattern recursively within a folder.
    files = dir(fullfile(folderPath, pattern)); % Find files matching the pattern
    files = files(~[files.isdir]); % Filter out directories
end

function fileNames = extractFileNames(files)
    % extractFileNames
    % Extract filenames without extensions from a list of file structures.
    [~, fileNames, ~] = cellfun(@fileparts, {files.name}, 'UniformOutput', false);
    fileNames = string(fileNames);
end

function filteredFiles = filterFilesByNames(files, names)
    % filterFilesByNames
    % Filter a list of file structures to include only those with specified names.
    filteredFiles = files(ismember(extractFileNames(files), names));
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
