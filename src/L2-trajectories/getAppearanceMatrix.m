function [ appearanceMatrix ] = getAppearanceMatrix(featureVectors, threshold )

% Computes the appearance affinity matrix
%{
featureVectors = featureVectors(indices);
threshold = params.threshold;
%}

features = double(cell2mat(featureVectors'));
dist = pdist2(features, features);
appearanceMatrix = (threshold - dist)/ threshold;


