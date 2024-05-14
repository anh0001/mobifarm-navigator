function model = modelBuild()
    % Builds a multi-sensor fusion deep learning architecture for mobile robotics.
    disp('Building the deep learning architecture...');

    %% RGB Image Processing Branch
    % Input: RGB images [480x640x3]
    imgInput = imageInputLayer([480 640 3], 'Name', 'image_input', 'Normalization', 'none');
    
    % Define CNN layers for RGB Images
    imgLayers = [
        convolution2dLayer(3, 16, 'Padding', 'same', 'Name', 'conv_img1', ...
            'WeightsInitializer', 'narrow-normal', 'Weights', randn([3 3 3 16]), ...
            'Bias', zeros(1, 1, 16)),
        batchNormalizationLayer('Name', 'bn_img1', 'TrainedMean', zeros(16, 1), 'TrainedVariance', ...
                ones(16, 1), 'Offset', zeros(16, 1), 'Scale', ones(16, 1)),
        reluLayer('Name', 'relu_img1'),
        maxPooling2dLayer(2, 'Stride', 2, 'Name', 'maxpool_img1'),
        
        convolution2dLayer(3, 32, 'Padding', 'same', 'Name', 'conv_img2', ...
            'WeightsInitializer', 'narrow-normal', 'Weights', randn([3 3 16 32]), ...
            'Bias', zeros(1, 1, 32)),
        batchNormalizationLayer('Name', 'bn_img2', 'TrainedMean', zeros(32, 1), 'TrainedVariance', ...
                ones(32, 1), 'Offset', zeros(32, 1), 'Scale', ones(32, 1)),
        reluLayer('Name', 'relu_img2'),
        maxPooling2dLayer(2, 'Stride', 2, 'Name', 'maxpool_img2'),

        convolution2dLayer(3, 64, 'Padding', 'same', 'Name', 'conv_img3', ...
            'WeightsInitializer', 'narrow-normal', 'Weights', randn([3 3 32 64]), ...
            'Bias', zeros(1, 1, 64)),
        batchNormalizationLayer('Name', 'bn_img3', 'TrainedMean', zeros(64, 1), 'TrainedVariance', ...
                ones(64, 1), 'Offset', zeros(64, 1), 'Scale', ones(64, 1)),
        reluLayer('Name', 'relu_img3'),
        globalAveragePooling2dLayer('Name', 'gap_img')
    ];

    %% Point Cloud Processing Branch
    % Input: Point cloud data [Nx3]
    pcInput = sequenceInputLayer(3, 'Name', 'pc_input', 'Normalization', 'none');

    % Define network layers for Point Clouds
    inputSize = 3;

    pcLayers = [
        lstmLayer(128, 'Name', 'lstm_pc1', 'InputWeightsInitializer', 'narrow-normal', ...
            'InputWeights', randn([512 inputSize]), ...
            'RecurrentWeightsInitializer', 'narrow-normal', 'RecurrentWeights', randn([512 128]), ...
            'BiasInitializer', 'narrow-normal', 'Bias', zeros(512, 1)),
        fullyConnectedLayer(256, 'Name', 'fc_pc1', ...
            'WeightsInitializer', 'narrow-normal', 'Weights', randn([256 128]), ...
            'Bias', zeros(256, 1)),
        reluLayer('Name', 'relu_pc1'),
        dropoutLayer(0.3, 'Name', 'dropout_pc1'),
        fullyConnectedLayer(128, 'Name', 'fc_pc2',...
            'WeightsInitializer', 'narrow-normal', 'Weights', randn([128 256]), ...
            'Bias', zeros(128, 1))
    ];

    %% LiDAR Distance Map Processing Branch
    % Input: LiDAR distance maps [320x640x1]
    lidarInput = imageInputLayer([320 640 1], 'Name', 'lidar_input', 'Normalization', 'none');
    
    % Define CNN layers for LiDAR Distance Maps
    lidarLayers = [
        convolution2dLayer(3, 16, 'Padding', 'same', 'Name', 'conv_lidar1', ...
            'WeightsInitializer', 'narrow-normal', 'Weights', randn([3 3 1 16]), ...
            'Bias', zeros(1, 1, 16)),
        batchNormalizationLayer('Name', 'bn_lidar1', 'TrainedMean', zeros(16, 1), 'TrainedVariance', ...
                ones(16, 1), 'Offset', zeros(16, 1), 'Scale', ones(16, 1)),
        reluLayer('Name', 'relu_lidar1'),
        maxPooling2dLayer(2, 'Stride', 2, 'Name', 'maxpool_lidar1'),

        convolution2dLayer(3, 32, 'Padding', 'same', 'Name', 'conv_lidar2', ...
            'WeightsInitializer', 'narrow-normal', 'Weights', randn([3 3 16 32]), ...
            'Bias', zeros(1, 1, 32)),
        batchNormalizationLayer('Name', 'bn_lidar2', 'TrainedMean', zeros(32, 1), 'TrainedVariance', ...
                ones(32, 1), 'Offset', zeros(32, 1), 'Scale', ones(32, 1)),
        reluLayer('Name', 'relu_lidar2'),
        globalAveragePooling2dLayer('Name', 'gap_lidar')
    ];

    %% Fusion and Prediction Layer
    fusionLayer = depthConcatenationLayer(3, 'Name', 'fusion_concat');
    finalLayers = [
        fullyConnectedLayer(256, 'Name', 'fc_final1', 'WeightsInitializer', 'narrow-normal', ...
            'Weights', randn([256 256]), 'Bias', zeros(256, 1)),
        reluLayer('Name', 'relu_final1'),
        dropoutLayer(0.5, 'Name', 'dropout_final'),
        fullyConnectedLayer(2, 'Name', 'output', ...
            'WeightsInitializer', 'narrow-normal', 'Weights', randn([2 256]), ...
            'Bias', zeros(2, 1)),  % Predicting linear and angular velocities
        regressionLayer('Name', 'regressionoutput')
    ];

    %% Assemble Layer Graph
    lgraph = layerGraph();

    % Add Input Layers separately
    lgraph = addLayers(lgraph, imgInput);
    lgraph = addLayers(lgraph, pcInput);
    lgraph = addLayers(lgraph, lidarInput);

    % Add Other Layers
    lgraph = addLayers(lgraph, imgLayers);
    lgraph = addLayers(lgraph, pcLayers);
    lgraph = addLayers(lgraph, lidarLayers);
    lgraph = addLayers(lgraph, fusionLayer);
    lgraph = addLayers(lgraph, finalLayers);

    % Flatten the output from the globalAveragePooling layers if not already a single vector
    flattenImg = flattenLayer('Name', 'flatten_img');
    flattenLidar = flattenLayer('Name', 'flatten_lidar');
    % Add these layers to the graph
    lgraph = addLayers(lgraph, flattenImg);
    lgraph = addLayers(lgraph, flattenLidar);

    % Connect Layers
    lgraph = connectLayers(lgraph, 'image_input', 'conv_img1');
    lgraph = connectLayers(lgraph, 'pc_input', 'lstm_pc1');
    lgraph = connectLayers(lgraph, 'lidar_input', 'conv_lidar1');

    % Connect these layers after the globalAveragePooling layers
    lgraph = connectLayers(lgraph, 'gap_img', 'flatten_img');
    lgraph = connectLayers(lgraph, 'gap_lidar', 'flatten_lidar');

    % Now connect the output of flatten layers and the output of the last fully connected layer in the point cloud branch to the fusion layer
    lgraph = connectLayers(lgraph, 'flatten_img', 'fusion_concat/in1');
    lgraph = connectLayers(lgraph, 'fc_pc2', 'fusion_concat/in2');
    lgraph = connectLayers(lgraph, 'flatten_lidar', 'fusion_concat/in3');

    lgraph = connectLayers(lgraph, 'fusion_concat', 'fc_final1');

    %% Finalize and Output the Model
    model = assembleNetwork(lgraph);
    disp('Model building complete.');
end