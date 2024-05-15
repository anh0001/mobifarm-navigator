function [dataTrain, dataVal] = dataPrep()
    % dataPrep.m
    % This function loads and preprocesses data from the 'data/raw' directory
    % and organizes it into training and validation datasets stored in 'data/processed'.

    disp('Starting data preparation...');

    %% Directories
    rawDir = 'data/raw/';
    processedDir = 'data/processed/';

    %% Load and Preprocess Training Data
    trainDir = fullfile(processedDir, 'train');
    dataTrain = preprocessData(trainDir);

    %% Load and Preprocess Validation Data
    valDir = fullfile(processedDir, 'val');
    dataVal = preprocessData(valDir);

    disp('Data preparation completed.');

end

function combinedDS = preprocessData(folderPath)
    % This function processes data from a specific folder, performing operations
    % such as loading images, labels, point cloud data, and maps.
    disp(['Processing data in: ', folderPath]);

    % Find all relevant files recursively
    imgFiles = findFilesRecursively(folderPath, '**/cam_front/*.jpg');
    labelFiles = findFilesRecursively(folderPath, '**/labels/*.txt');
    lidarFiles = findFilesRecursively(folderPath, '**/realtime_map/*.jpg');
    pcdFiles = findFilesRecursively(folderPath, '**/pcd_preprocessed/*.npy');

    % Extract full file paths
    imgFiles = fullfile({imgFiles.folder}, {imgFiles.name});
    labelFiles = fullfile({labelFiles.folder}, {labelFiles.name});
    lidarFiles = fullfile({lidarFiles.folder}, {lidarFiles.name});
    pcdFiles = fullfile({pcdFiles.folder}, {pcdFiles.name});

    % Load Camera Images using imageDatastore
    imgDS = imageDatastore(imgFiles, 'FileExtensions', {'.jpg'}, 'ReadFcn', @readImage);

    % Load Labels using fileDatastore
    labelDS = fileDatastore(labelFiles, 'ReadFcn', @readmatrix, 'FileExtensions', {'.txt'});

    % Load Processed Point Cloud Data using fileDatastore
    pcdDS = fileDatastore(pcdFiles, 'ReadFcn', @readNPY, 'FileExtensions', {'.npy'});

    % Load Real-time Maps using imageDatastore
    lidarDS = imageDatastore(lidarFiles, 'FileExtensions', {'.jpg'}, 'ReadFcn', @readImage);

    % Combine the datastores into a single datastore
    combinedDS = combine(imgDS, lidarDS, pcdDS, labelDS);
end

function files = findFilesRecursively(folderPath, pattern)
    files = dir(fullfile(folderPath, pattern));
    files = files(~[files.isdir]);
end

function data = readImage(filename)
    % Custom read function for imageDatastore
    data = imread(filename);
end

function data = readNPY(filename)
    % Custom read function for fileDatastore to load .npy files
    data = readNPY(filename); % Assuming you have a readNPY function available
end