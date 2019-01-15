function estimatedVelocities = estimateVelocities(originalDetections, startFrame, endFrame, nearestNeighbors, speedLimit)
% This function estimates the velocity of a detection by calculating the component-wise
% median of velocities required to reach a specified number of nearest neighbors.
% Neighbors that exceed a specified speed limit are not considered.
%{
nearestNeighbors = params.nearest_neighbors;
speedLimit = params.speed_limit;
%}

% Find detections in search range
searchRangeMask    = intervalSearch(originalDetections(:,1), startFrame - nearestNeighbors, endFrame+ nearestNeighbors);
searchRangeCenters = getBoundingBoxCenters(originalDetections(searchRangeMask, 3:6));
searchRangeFrames  = originalDetections(searchRangeMask, 1);
detectionIndices   = intervalSearch(searchRangeFrames, startFrame, endFrame);

% Compute all pairwise distances 
% -- pdist2 is to calucute each detection with each detection's distance
% -- use sqrt((x1-x2)^2 + (y1-y2)^2)
pairDistance        = pdist2(searchRangeCenters,searchRangeCenters);
numDetections       = length(detectionIndices);
estimatedVelocities = zeros(numDetections,2);

% Estimate the velocity of each detection
for i = 1:numDetections
    
    currentDetectionIndex = detectionIndices(i);
    
    velocities = [];
    currentFrame = searchRangeFrames(currentDetectionIndex);
    
    % For each time instant in a small time neighborhood find the nearest detection in space
    for frame = currentFrame-nearestNeighbors:currentFrame+nearestNeighbors
        
        % Skip original frame
        if abs(currentFrame-frame) <= 0
            continue;
        end
        
        detectionsAtThisTimeInstant = searchRangeFrames == frame;
        
        % Skip if no detections in the current frame
        if sum(detectionsAtThisTimeInstant) == 0
            continue;
        end
        
        distancesAtThisTimeInstant = pairDistance(currentDetectionIndex,:);
        distancesAtThisTimeInstant(detectionsAtThisTimeInstant==0) = inf;
        
        % Find detection closest to the current detection
        % -- 去算當前的i與每一個時間點frame中 最小距離的速度
        % -- 速度算法為 x = x2 - x1, y = y2-y1;
        [~, targetDetectionIndex] = min(distancesAtThisTimeInstant);
        estimatedVelocity = searchRangeCenters(targetDetectionIndex,:) - searchRangeCenters(currentDetectionIndex,:);
        estimatedVelocity = estimatedVelocity / (searchRangeFrames(targetDetectionIndex) - searchRangeFrames(currentDetectionIndex));
        
        %estimatedVelocity = [double(estimatedVelocity(1)), double(estimatedVelocity(2))]
        % Check if speed limit is violated
        estimatedSpeed = norm(estimatedVelocity);
        if estimatedSpeed > speedLimit
            continue;
        end
        
        % Update velocity estimates
        % -- 把當前的i與其他frame最小距離的所有速度結果都存入這裡
        velocities = [velocities; estimatedVelocity];
        
    end
    
    if isempty(velocities)
        velocities = [0 0];
    end
    
    % Estimate the velocity
    % -- 最後將結果取x平均 與y 平均
    estimatedVelocities(i,1) = mean(velocities(:,1));
    estimatedVelocities(i,2) = mean(velocities(:,2));
    
    % -- return 2d array, mean x-velocity and y-velocity
end




