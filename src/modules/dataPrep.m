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

        % Ensure that 'images' is a column vector
        images = images(:);

        % Concatenate 'images' to 'data.images'
        if isempty(data.images)
            data.images = images;
        else
            % Ensure that 'data.images' and 'images' have the same size in the second dimension
            assert(size(data.images, 2) == size(images, 2), 'Size mismatch between ''data.images'' and ''images''');
            data.images = [data.images; images];
        end

        % Load Labels
        labelFiles = dir(fullfile(currentFolder, 'labels', '*.txt'));
        labels = cell(1, numel(labelFiles));
        for j = 1:numel(labelFiles)
            % Read label data from text file, each row contains a linear and angular velocity
            labels{j} = readmatrix(fullfile(labelFiles(j).folder, labelFiles(j).name));
        end

        % Ensure that 'labels' is a column vector
        labels = labels(:);

        % Concatenate 'labels' to 'data.labels'
        if isempty(data.labels)
            data.labels = labels;
        else
            % Ensure that 'data.labels' and 'labels' have the same size in the second dimension
            assert(size(data.labels, 2) == size(labels, 2), 'Size mismatch between ''data.labels'' and ''labels''');
            data.labels = [data.labels; labels];
        end

        % Load Processed Point Cloud Data
        pcdFiles = dir(fullfile(currentFolder, 'pcd_preprocessed', '*.npy'));
        pointClouds = cell(1, numel(pcdFiles));
        for j = 1:numel(pcdFiles)
            % Read point cloud data from .npy file
            pointClouds{j} = readNPY(fullfile(pcdFiles(j).folder, pcdFiles(j).name));
        end

        % Ensure that 'pointClouds' is a column vector
        pointClouds = pointClouds(:);

        % Concatenate 'pointClouds' to 'data.pointClouds'
        if isempty(data.pointClouds)
            data.pointClouds = pointClouds;
        else
            % Ensure that 'data.pointClouds' and 'pointClouds' have the same size in the second dimension
            assert(size(data.pointClouds, 2) == size(pointClouds, 2), 'Size mismatch between ''data.pointClouds'' and ''pointClouds''');
            data.pointClouds = [data.pointClouds; pointClouds];
        end

        % Load Real-time Maps
        mapFiles = dir(fullfile(currentFolder, 'realtime_map', '*.jpg'));
        maps = cell(1, numel(mapFiles));
        for j = 1:numel(mapFiles)
            % Load map data from .jpg file
            maps{j} = imread(fullfile(mapFiles(j).folder, mapFiles(j).name));
        end

        % Ensure that 'maps' is a column vector
        maps = maps(:);

        % Concatenate 'maps' to 'data.maps'
        if isempty(data.maps)
            data.maps = maps;
        else
            % Ensure that 'data.maps' and 'maps' have the same size in the second dimension
            assert(size(data.maps, 2) == size(maps, 2), 'Size mismatch between ''data.maps'' and ''maps''');
            data.maps = [data.maps; maps];
        end
    end
end
