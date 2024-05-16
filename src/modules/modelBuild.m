%% Create Deep Learning Network Architecture with Pretrained Parameters
% Script for creating the layers for a deep learning network with the following
% properties:
%%
%
%  Number of layers: 152
%  Number of connections: 167
%  Pretrained parameters file: /home/mobi18/Codes/mobifarm-navigator/params_2024_05_14__15_23_36.mat
%
%%
% Run the script to create the layers in the workspace variable |net|.
%
% To learn more, see <matlab:helpview('deeplearning','generate_matlab_code')
% Generate MATLAB Code From Deep Network Designer>.
%
% Auto-generated by MATLAB on 2024/05/14 15:23:54
%% Load Network Parameters
% Load network parameters like weights, biases, or layers unsupported for network
% code generation from the stored parameters file.
function model = modelBuild()
params = load("params_2024_05_14__15_23_36.mat");
%% Create dlnetwork
% Create the dlnetwork variable to contain the network layers.

net = dlnetwork;
%% Add Layer Branches
% Add branches to the dlnetwork. Each branch is a linear array of layers.

tempNet = [
    imageInputLayer([480 640 3],"Name","image_input","Mean",params.image_input.Mean)
    convolution2dLayer([3 3],64,"Name","conv1","Stride",[2 2],"Bias",params.conv1.Bias,"Weights",params.conv1.Weights)
    reluLayer("Name","relu_conv1")
    maxPooling2dLayer([3 3],"Name","pool1","Stride",[2 2])
    convolution2dLayer([1 1],16,"Name","fire2-squeeze1x1","Bias",params.fire2_squeeze1x1.Bias,"Weights",params.fire2_squeeze1x1.Weights)
    reluLayer("Name","fire2-relu_squeeze1x1")];
net = addLayers(net,tempNet);

tempNet = [
    convolution2dLayer([1 1],64,"Name","fire2-expand1x1","Bias",params.fire2_expand1x1.Bias,"Weights",params.fire2_expand1x1.Weights)
    reluLayer("Name","fire2-relu_expand1x1")];
net = addLayers(net,tempNet);

tempNet = [
    convolution2dLayer([3 3],64,"Name","fire2-expand3x3","Padding",[1 1 1 1],"Bias",params.fire2_expand3x3.Bias,"Weights",params.fire2_expand3x3.Weights)
    reluLayer("Name","fire2-relu_expand3x3")];
net = addLayers(net,tempNet);

tempNet = [
    depthConcatenationLayer(2,"Name","fire2-concat")
    convolution2dLayer([1 1],16,"Name","fire3-squeeze1x1","Bias",params.fire3_squeeze1x1.Bias,"Weights",params.fire3_squeeze1x1.Weights)
    reluLayer("Name","fire3-relu_squeeze1x1")];
net = addLayers(net,tempNet);

tempNet = [
    convolution2dLayer([1 1],64,"Name","fire3-expand1x1","Bias",params.fire3_expand1x1.Bias,"Weights",params.fire3_expand1x1.Weights)
    reluLayer("Name","fire3-relu_expand1x1")];
net = addLayers(net,tempNet);

tempNet = [
    convolution2dLayer([3 3],64,"Name","fire3-expand3x3","Padding",[1 1 1 1],"Bias",params.fire3_expand3x3.Bias,"Weights",params.fire3_expand3x3.Weights)
    reluLayer("Name","fire3-relu_expand3x3")];
net = addLayers(net,tempNet);

tempNet = [
    depthConcatenationLayer(2,"Name","fire3-concat")
    maxPooling2dLayer([3 3],"Name","pool3","Padding",[0 1 0 1],"Stride",[2 2])
    convolution2dLayer([1 1],32,"Name","fire4-squeeze1x1","Bias",params.fire4_squeeze1x1.Bias,"Weights",params.fire4_squeeze1x1.Weights)
    reluLayer("Name","fire4-relu_squeeze1x1")];
net = addLayers(net,tempNet);

tempNet = [
    convolution2dLayer([1 1],128,"Name","fire4-expand1x1","Bias",params.fire4_expand1x1.Bias,"Weights",params.fire4_expand1x1.Weights)
    reluLayer("Name","fire4-relu_expand1x1")];
net = addLayers(net,tempNet);

tempNet = [
    convolution2dLayer([3 3],128,"Name","fire4-expand3x3","Padding",[1 1 1 1],"Bias",params.fire4_expand3x3.Bias,"Weights",params.fire4_expand3x3.Weights)
    reluLayer("Name","fire4-relu_expand3x3")];
net = addLayers(net,tempNet);

tempNet = [
    depthConcatenationLayer(2,"Name","fire4-concat")
    convolution2dLayer([1 1],32,"Name","fire5-squeeze1x1","Bias",params.fire5_squeeze1x1.Bias,"Weights",params.fire5_squeeze1x1.Weights)
    reluLayer("Name","fire5-relu_squeeze1x1")];
net = addLayers(net,tempNet);

tempNet = [
    convolution2dLayer([1 1],128,"Name","fire5-expand1x1","Bias",params.fire5_expand1x1.Bias,"Weights",params.fire5_expand1x1.Weights)
    reluLayer("Name","fire5-relu_expand1x1")];
net = addLayers(net,tempNet);

tempNet = [
    convolution2dLayer([3 3],128,"Name","fire5-expand3x3","Padding",[1 1 1 1],"Bias",params.fire5_expand3x3.Bias,"Weights",params.fire5_expand3x3.Weights)
    reluLayer("Name","fire5-relu_expand3x3")];
net = addLayers(net,tempNet);

tempNet = [
    depthConcatenationLayer(2,"Name","fire5-concat")
    maxPooling2dLayer([3 3],"Name","pool5","Padding",[0 1 0 1],"Stride",[2 2])
    convolution2dLayer([1 1],48,"Name","fire6-squeeze1x1","Bias",params.fire6_squeeze1x1.Bias,"Weights",params.fire6_squeeze1x1.Weights)
    reluLayer("Name","fire6-relu_squeeze1x1")];
net = addLayers(net,tempNet);

tempNet = [
    convolution2dLayer([1 1],192,"Name","fire6-expand1x1","Bias",params.fire6_expand1x1.Bias,"Weights",params.fire6_expand1x1.Weights)
    reluLayer("Name","fire6-relu_expand1x1")];
net = addLayers(net,tempNet);

tempNet = [
    convolution2dLayer([3 3],192,"Name","fire6-expand3x3","Padding",[1 1 1 1],"Bias",params.fire6_expand3x3.Bias,"Weights",params.fire6_expand3x3.Weights)
    reluLayer("Name","fire6-relu_expand3x3")];
net = addLayers(net,tempNet);

tempNet = [
    depthConcatenationLayer(2,"Name","fire6-concat")
    convolution2dLayer([1 1],48,"Name","fire7-squeeze1x1","Bias",params.fire7_squeeze1x1.Bias,"Weights",params.fire7_squeeze1x1.Weights)
    reluLayer("Name","fire7-relu_squeeze1x1")];
net = addLayers(net,tempNet);

tempNet = [
    convolution2dLayer([1 1],192,"Name","fire7-expand1x1","Bias",params.fire7_expand1x1.Bias,"Weights",params.fire7_expand1x1.Weights)
    reluLayer("Name","fire7-relu_expand1x1")];
net = addLayers(net,tempNet);

tempNet = [
    convolution2dLayer([3 3],192,"Name","fire7-expand3x3","Padding",[1 1 1 1],"Bias",params.fire7_expand3x3.Bias,"Weights",params.fire7_expand3x3.Weights)
    reluLayer("Name","fire7-relu_expand3x3")];
net = addLayers(net,tempNet);

tempNet = [
    depthConcatenationLayer(2,"Name","fire7-concat")
    convolution2dLayer([1 1],64,"Name","fire8-squeeze1x1","Bias",params.fire8_squeeze1x1.Bias,"Weights",params.fire8_squeeze1x1.Weights)
    reluLayer("Name","fire8-relu_squeeze1x1")];
net = addLayers(net,tempNet);

tempNet = [
    convolution2dLayer([1 1],256,"Name","fire8-expand1x1","Bias",params.fire8_expand1x1.Bias,"Weights",params.fire8_expand1x1.Weights)
    reluLayer("Name","fire8-relu_expand1x1")];
net = addLayers(net,tempNet);

tempNet = [
    convolution2dLayer([3 3],256,"Name","fire8-expand3x3","Padding",[1 1 1 1],"Bias",params.fire8_expand3x3.Bias,"Weights",params.fire8_expand3x3.Weights)
    reluLayer("Name","fire8-relu_expand3x3")];
net = addLayers(net,tempNet);

tempNet = [
    depthConcatenationLayer(2,"Name","fire8-concat")
    convolution2dLayer([1 1],64,"Name","fire9-squeeze1x1","Bias",params.fire9_squeeze1x1.Bias,"Weights",params.fire9_squeeze1x1.Weights)
    reluLayer("Name","fire9-relu_squeeze1x1")];
net = addLayers(net,tempNet);

tempNet = [
    convolution2dLayer([1 1],256,"Name","fire9-expand1x1","Bias",params.fire9_expand1x1.Bias,"Weights",params.fire9_expand1x1.Weights)
    reluLayer("Name","fire9-relu_expand1x1")];
net = addLayers(net,tempNet);

tempNet = [
    convolution2dLayer([3 3],256,"Name","fire9-expand3x3","Padding",[1 1 1 1],"Bias",params.fire9_expand3x3.Bias,"Weights",params.fire9_expand3x3.Weights)
    reluLayer("Name","fire9-relu_expand3x3")];
net = addLayers(net,tempNet);

tempNet = [
    depthConcatenationLayer(2,"Name","fire9-concat")
    dropoutLayer(0.5,"Name","drop9")
    convolution2dLayer([1 1],1000,"Name","conv10","Bias",params.conv10.Bias,"Weights",params.conv10.Weights)
    reluLayer("Name","relu_conv10")
    globalAveragePooling2dLayer("Name","pool10")
    flattenLayer("Name","prob_flatten")];
net = addLayers(net,tempNet);

tempNet = [
    imageInputLayer([320 640 3],"Name","lidar_input","Mean",params.lidar_input.Mean)
    convolution2dLayer([3 3],64,"Name","conv1_1","Stride",[2 2],"Bias",params.conv1_1.Bias,"Weights",params.conv1_1.Weights)
    reluLayer("Name","relu_conv1_1")
    maxPooling2dLayer([3 3],"Name","pool1_1","Stride",[2 2])
    convolution2dLayer([1 1],16,"Name","fire2-squeeze1x1_1","Bias",params.fire2_squeeze1x1_1.Bias,"Weights",params.fire2_squeeze1x1_1.Weights)
    reluLayer("Name","fire2-relu_squeeze1x1_1")];
net = addLayers(net,tempNet);

tempNet = [
    convolution2dLayer([1 1],64,"Name","fire2-expand1x1_1","Bias",params.fire2_expand1x1_1.Bias,"Weights",params.fire2_expand1x1_1.Weights)
    reluLayer("Name","fire2-relu_expand1x1_1")];
net = addLayers(net,tempNet);

tempNet = [
    convolution2dLayer([3 3],64,"Name","fire2-expand3x3_1","Padding",[1 1 1 1],"Bias",params.fire2_expand3x3_1.Bias,"Weights",params.fire2_expand3x3_1.Weights)
    reluLayer("Name","fire2-relu_expand3x3_1")];
net = addLayers(net,tempNet);

tempNet = [
    depthConcatenationLayer(2,"Name","fire2-concat_1")
    convolution2dLayer([1 1],16,"Name","fire3-squeeze1x1_1","Bias",params.fire3_squeeze1x1_1.Bias,"Weights",params.fire3_squeeze1x1_1.Weights)
    reluLayer("Name","fire3-relu_squeeze1x1_1")];
net = addLayers(net,tempNet);

tempNet = [
    convolution2dLayer([1 1],64,"Name","fire3-expand1x1_1","Bias",params.fire3_expand1x1_1.Bias,"Weights",params.fire3_expand1x1_1.Weights)
    reluLayer("Name","fire3-relu_expand1x1_1")];
net = addLayers(net,tempNet);

tempNet = [
    convolution2dLayer([3 3],64,"Name","fire3-expand3x3_1","Padding",[1 1 1 1],"Bias",params.fire3_expand3x3_1.Bias,"Weights",params.fire3_expand3x3_1.Weights)
    reluLayer("Name","fire3-relu_expand3x3_1")];
net = addLayers(net,tempNet);

tempNet = [
    depthConcatenationLayer(2,"Name","fire3-concat_1")
    maxPooling2dLayer([3 3],"Name","pool3_1","Padding",[0 1 0 1],"Stride",[2 2])
    convolution2dLayer([1 1],32,"Name","fire4-squeeze1x1_1","Bias",params.fire4_squeeze1x1_1.Bias,"Weights",params.fire4_squeeze1x1_1.Weights)
    reluLayer("Name","fire4-relu_squeeze1x1_1")];
net = addLayers(net,tempNet);

tempNet = [
    convolution2dLayer([1 1],128,"Name","fire4-expand1x1_1","Bias",params.fire4_expand1x1_1.Bias,"Weights",params.fire4_expand1x1_1.Weights)
    reluLayer("Name","fire4-relu_expand1x1_1")];
net = addLayers(net,tempNet);

tempNet = [
    convolution2dLayer([3 3],128,"Name","fire4-expand3x3_1","Padding",[1 1 1 1],"Bias",params.fire4_expand3x3_1.Bias,"Weights",params.fire4_expand3x3_1.Weights)
    reluLayer("Name","fire4-relu_expand3x3_1")];
net = addLayers(net,tempNet);

tempNet = [
    depthConcatenationLayer(2,"Name","fire4-concat_1")
    convolution2dLayer([1 1],32,"Name","fire5-squeeze1x1_1","Bias",params.fire5_squeeze1x1_1.Bias,"Weights",params.fire5_squeeze1x1_1.Weights)
    reluLayer("Name","fire5-relu_squeeze1x1_1")];
net = addLayers(net,tempNet);

tempNet = [
    convolution2dLayer([1 1],128,"Name","fire5-expand1x1_1","Bias",params.fire5_expand1x1_1.Bias,"Weights",params.fire5_expand1x1_1.Weights)
    reluLayer("Name","fire5-relu_expand1x1_1")];
net = addLayers(net,tempNet);

tempNet = [
    convolution2dLayer([3 3],128,"Name","fire5-expand3x3_1","Padding",[1 1 1 1],"Bias",params.fire5_expand3x3_1.Bias,"Weights",params.fire5_expand3x3_1.Weights)
    reluLayer("Name","fire5-relu_expand3x3_1")];
net = addLayers(net,tempNet);

tempNet = [
    depthConcatenationLayer(2,"Name","fire5-concat_1")
    maxPooling2dLayer([3 3],"Name","pool5_1","Padding",[0 1 0 1],"Stride",[2 2])
    convolution2dLayer([1 1],48,"Name","fire6-squeeze1x1_1","Bias",params.fire6_squeeze1x1_1.Bias,"Weights",params.fire6_squeeze1x1_1.Weights)
    reluLayer("Name","fire6-relu_squeeze1x1_1")];
net = addLayers(net,tempNet);

tempNet = [
    convolution2dLayer([1 1],192,"Name","fire6-expand1x1_1","Bias",params.fire6_expand1x1_1.Bias,"Weights",params.fire6_expand1x1_1.Weights)
    reluLayer("Name","fire6-relu_expand1x1_1")];
net = addLayers(net,tempNet);

tempNet = [
    convolution2dLayer([3 3],192,"Name","fire6-expand3x3_1","Padding",[1 1 1 1],"Bias",params.fire6_expand3x3_1.Bias,"Weights",params.fire6_expand3x3_1.Weights)
    reluLayer("Name","fire6-relu_expand3x3_1")];
net = addLayers(net,tempNet);

tempNet = [
    depthConcatenationLayer(2,"Name","fire6-concat_1")
    convolution2dLayer([1 1],48,"Name","fire7-squeeze1x1_1","Bias",params.fire7_squeeze1x1_1.Bias,"Weights",params.fire7_squeeze1x1_1.Weights)
    reluLayer("Name","fire7-relu_squeeze1x1_1")];
net = addLayers(net,tempNet);

tempNet = [
    convolution2dLayer([1 1],192,"Name","fire7-expand1x1_1","Bias",params.fire7_expand1x1_1.Bias,"Weights",params.fire7_expand1x1_1.Weights)
    reluLayer("Name","fire7-relu_expand1x1_1")];
net = addLayers(net,tempNet);

tempNet = [
    convolution2dLayer([3 3],192,"Name","fire7-expand3x3_1","Padding",[1 1 1 1],"Bias",params.fire7_expand3x3_1.Bias,"Weights",params.fire7_expand3x3_1.Weights)
    reluLayer("Name","fire7-relu_expand3x3_1")];
net = addLayers(net,tempNet);

tempNet = [
    depthConcatenationLayer(2,"Name","fire7-concat_1")
    convolution2dLayer([1 1],64,"Name","fire8-squeeze1x1_1","Bias",params.fire8_squeeze1x1_1.Bias,"Weights",params.fire8_squeeze1x1_1.Weights)
    reluLayer("Name","fire8-relu_squeeze1x1_1")];
net = addLayers(net,tempNet);

tempNet = [
    convolution2dLayer([1 1],256,"Name","fire8-expand1x1_1","Bias",params.fire8_expand1x1_1.Bias,"Weights",params.fire8_expand1x1_1.Weights)
    reluLayer("Name","fire8-relu_expand1x1_1")];
net = addLayers(net,tempNet);

tempNet = [
    convolution2dLayer([3 3],256,"Name","fire8-expand3x3_1","Padding",[1 1 1 1],"Bias",params.fire8_expand3x3_1.Bias,"Weights",params.fire8_expand3x3_1.Weights)
    reluLayer("Name","fire8-relu_expand3x3_1")];
net = addLayers(net,tempNet);

tempNet = [
    depthConcatenationLayer(2,"Name","fire8-concat_1")
    convolution2dLayer([1 1],64,"Name","fire9-squeeze1x1_1","Bias",params.fire9_squeeze1x1_1.Bias,"Weights",params.fire9_squeeze1x1_1.Weights)
    reluLayer("Name","fire9-relu_squeeze1x1_1")];
net = addLayers(net,tempNet);

tempNet = [
    convolution2dLayer([1 1],256,"Name","fire9-expand1x1_1","Bias",params.fire9_expand1x1_1.Bias,"Weights",params.fire9_expand1x1_1.Weights)
    reluLayer("Name","fire9-relu_expand1x1_1")];
net = addLayers(net,tempNet);

tempNet = [
    convolution2dLayer([3 3],256,"Name","fire9-expand3x3_1","Padding",[1 1 1 1],"Bias",params.fire9_expand3x3_1.Bias,"Weights",params.fire9_expand3x3_1.Weights)
    reluLayer("Name","fire9-relu_expand3x3_1")];
net = addLayers(net,tempNet);

tempNet = [
    depthConcatenationLayer(2,"Name","fire9-concat_1")
    dropoutLayer(0.5,"Name","drop9_1")
    convolution2dLayer([1 1],1000,"Name","conv10_1","Bias",params.conv10_1.Bias,"Weights",params.conv10_1.Weights)
    reluLayer("Name","relu_conv10_1")
    globalAveragePooling2dLayer("Name","pool10_1")
    flattenLayer("Name","prob_flatten_1")];
net = addLayers(net,tempNet);

tempNet = [
    imageInputLayer([61440 3 1],"Name","pc_input")
    convolution2dLayer([1 3],64,"Name","conv","Padding","same","WeightsInitializer","narrow-normal")
    batchNormalizationLayer("Name","batchnorm")
    reluLayer("Name","relu_1")
    convolution2dLayer([1 1],128,"Name","conv_1","Padding","same","WeightsInitializer","narrow-normal")
    batchNormalizationLayer("Name","batchnorm_1")
    reluLayer("Name","relu_2")
    convolution2dLayer([1 1],256,"Name","conv_2","Padding","same","WeightsInitializer","narrow-normal")
    batchNormalizationLayer("Name","batchnorm_2")
    reluLayer("Name","relu_3")
    globalAveragePooling2dLayer("Name","gapool")
    flattenLayer("Name","prob_flatten_1_1")];
net = addLayers(net,tempNet);

tempNet = [
    depthConcatenationLayer(3,"Name","depthcat")
    fullyConnectedLayer(256,"Name","fc","WeightsInitializer","narrow-normal")
    reluLayer("Name","relu")
    dropoutLayer(0.5,"Name","dropout")
    fullyConnectedLayer(2,"Name","fc_1","WeightsInitializer","narrow-normal")];
net = addLayers(net,tempNet);

% clean up helper variable
clear tempNet;
%% Connect Layer Branches
% Connect all the branches of the network to create the network graph.

net = connectLayers(net,"fire2-relu_squeeze1x1","fire2-expand1x1");
net = connectLayers(net,"fire2-relu_squeeze1x1","fire2-expand3x3");
net = connectLayers(net,"fire2-relu_expand1x1","fire2-concat/in1");
net = connectLayers(net,"fire2-relu_expand3x3","fire2-concat/in2");
net = connectLayers(net,"fire3-relu_squeeze1x1","fire3-expand1x1");
net = connectLayers(net,"fire3-relu_squeeze1x1","fire3-expand3x3");
net = connectLayers(net,"fire3-relu_expand1x1","fire3-concat/in1");
net = connectLayers(net,"fire3-relu_expand3x3","fire3-concat/in2");
net = connectLayers(net,"fire4-relu_squeeze1x1","fire4-expand1x1");
net = connectLayers(net,"fire4-relu_squeeze1x1","fire4-expand3x3");
net = connectLayers(net,"fire4-relu_expand1x1","fire4-concat/in1");
net = connectLayers(net,"fire4-relu_expand3x3","fire4-concat/in2");
net = connectLayers(net,"fire5-relu_squeeze1x1","fire5-expand1x1");
net = connectLayers(net,"fire5-relu_squeeze1x1","fire5-expand3x3");
net = connectLayers(net,"fire5-relu_expand1x1","fire5-concat/in1");
net = connectLayers(net,"fire5-relu_expand3x3","fire5-concat/in2");
net = connectLayers(net,"fire6-relu_squeeze1x1","fire6-expand1x1");
net = connectLayers(net,"fire6-relu_squeeze1x1","fire6-expand3x3");
net = connectLayers(net,"fire6-relu_expand1x1","fire6-concat/in1");
net = connectLayers(net,"fire6-relu_expand3x3","fire6-concat/in2");
net = connectLayers(net,"fire7-relu_squeeze1x1","fire7-expand1x1");
net = connectLayers(net,"fire7-relu_squeeze1x1","fire7-expand3x3");
net = connectLayers(net,"fire7-relu_expand1x1","fire7-concat/in1");
net = connectLayers(net,"fire7-relu_expand3x3","fire7-concat/in2");
net = connectLayers(net,"fire8-relu_squeeze1x1","fire8-expand1x1");
net = connectLayers(net,"fire8-relu_squeeze1x1","fire8-expand3x3");
net = connectLayers(net,"fire8-relu_expand1x1","fire8-concat/in1");
net = connectLayers(net,"fire8-relu_expand3x3","fire8-concat/in2");
net = connectLayers(net,"fire9-relu_squeeze1x1","fire9-expand1x1");
net = connectLayers(net,"fire9-relu_squeeze1x1","fire9-expand3x3");
net = connectLayers(net,"fire9-relu_expand1x1","fire9-concat/in1");
net = connectLayers(net,"fire9-relu_expand3x3","fire9-concat/in2");
net = connectLayers(net,"prob_flatten","depthcat/in1");
net = connectLayers(net,"fire2-relu_squeeze1x1_1","fire2-expand1x1_1");
net = connectLayers(net,"fire2-relu_squeeze1x1_1","fire2-expand3x3_1");
net = connectLayers(net,"fire2-relu_expand1x1_1","fire2-concat_1/in1");
net = connectLayers(net,"fire2-relu_expand3x3_1","fire2-concat_1/in2");
net = connectLayers(net,"fire3-relu_squeeze1x1_1","fire3-expand1x1_1");
net = connectLayers(net,"fire3-relu_squeeze1x1_1","fire3-expand3x3_1");
net = connectLayers(net,"fire3-relu_expand1x1_1","fire3-concat_1/in1");
net = connectLayers(net,"fire3-relu_expand3x3_1","fire3-concat_1/in2");
net = connectLayers(net,"fire4-relu_squeeze1x1_1","fire4-expand1x1_1");
net = connectLayers(net,"fire4-relu_squeeze1x1_1","fire4-expand3x3_1");
net = connectLayers(net,"fire4-relu_expand1x1_1","fire4-concat_1/in1");
net = connectLayers(net,"fire4-relu_expand3x3_1","fire4-concat_1/in2");
net = connectLayers(net,"fire5-relu_squeeze1x1_1","fire5-expand1x1_1");
net = connectLayers(net,"fire5-relu_squeeze1x1_1","fire5-expand3x3_1");
net = connectLayers(net,"fire5-relu_expand1x1_1","fire5-concat_1/in1");
net = connectLayers(net,"fire5-relu_expand3x3_1","fire5-concat_1/in2");
net = connectLayers(net,"fire6-relu_squeeze1x1_1","fire6-expand1x1_1");
net = connectLayers(net,"fire6-relu_squeeze1x1_1","fire6-expand3x3_1");
net = connectLayers(net,"fire6-relu_expand1x1_1","fire6-concat_1/in1");
net = connectLayers(net,"fire6-relu_expand3x3_1","fire6-concat_1/in2");
net = connectLayers(net,"fire7-relu_squeeze1x1_1","fire7-expand1x1_1");
net = connectLayers(net,"fire7-relu_squeeze1x1_1","fire7-expand3x3_1");
net = connectLayers(net,"fire7-relu_expand1x1_1","fire7-concat_1/in1");
net = connectLayers(net,"fire7-relu_expand3x3_1","fire7-concat_1/in2");
net = connectLayers(net,"fire8-relu_squeeze1x1_1","fire8-expand1x1_1");
net = connectLayers(net,"fire8-relu_squeeze1x1_1","fire8-expand3x3_1");
net = connectLayers(net,"fire8-relu_expand1x1_1","fire8-concat_1/in1");
net = connectLayers(net,"fire8-relu_expand3x3_1","fire8-concat_1/in2");
net = connectLayers(net,"fire9-relu_squeeze1x1_1","fire9-expand1x1_1");
net = connectLayers(net,"fire9-relu_squeeze1x1_1","fire9-expand3x3_1");
net = connectLayers(net,"fire9-relu_expand1x1_1","fire9-concat_1/in1");
net = connectLayers(net,"fire9-relu_expand3x3_1","fire9-concat_1/in2");
net = connectLayers(net,"prob_flatten_1","depthcat/in2");
net = connectLayers(net,"prob_flatten_1_1","depthcat/in3");
net = initialize(net);

model = net;
%% Plot Layers

plot(net);
end