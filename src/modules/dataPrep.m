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

function data = preprocessData(folderPath)
    % This function processes data from a specific folder, performing operations
    % such as loading images, labels, point cloud data, and maps.
    
    data = struct;
    disp(['Processing data in: ', folderPath]);

    % Define data structure
    data.images = [];
    data.labels = [];
    data.pointClouds = [];
    data.maps = [];

    % Check each dataset folder within the directory
    datasetFolders = dir(fullfile(folderPath, 'data_*'));
    for i = 1:length(datasetFolders)
        currentFolder = fullfile(datasetFolders(i).folder, datasetFolders(i).name);
        disp(['Loading data from: ', currentFolder]);

        % Load Camera Images
        imgFiles = dir(fullfile(currentFolder, 'cam_front', '*.jpg'));
        images = cell(1, numel(imgFiles));
        for j = 1:numel(imgFiles)
            images{j} = imread(fullfile(imgFiles(j).folder, imgFiles(j).name));
        end
        data.images = [data.images; images];

        % Load Labels
        labelFiles = dir(fullfile(currentFolder, 'labels', '*.txt'));
        labels = cell(1, numel(labelFiles));
        for j = 1:numel(labelFiles)
            % Read label data from text file, each row contains a linear and angular velocity
            labels{j} = readmatrix(fullfile(labelFiles(j).folder, labelFiles(j).name));
        end
        data.labels = [data.labels; labels];

        % Load Processed Point Cloud Data
        pcdFiles = dir(fullfile(currentFolder, 'pcd_preprocessed', '*.npy'));
        pointClouds = cell(1, numel(pcdFiles));
        for j = 1:numel(pcdFiles)
            temp = load(fullfile(pcdFiles(j).folder, pcdFiles(j).name));
            pointClouds{j} = temp.pcd; % Assuming 'pcd' is the variable name in the .mat file
        end
        data.pointClouds = [data.pointClouds; pointClouds];

        % Load Real-time Maps
        mapFiles = dir(fullfile(currentFolder, 'realtime_map', '*.mat'));
        maps = cell(1, numel(mapFiles));
        for j = 1:numel(mapFiles)
            temp = load(fullfile(mapFiles(j).folder, mapFiles(j).name));
            maps{j} = temp.map; % Assuming 'map' is the variable name in the .mat file
        end
        data.maps = [data.maps; maps];
    end
end
