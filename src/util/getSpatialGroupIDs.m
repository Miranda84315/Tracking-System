function spatialGroupIDs = getSpatialGroupIDs(useGrouping, currentDetectionsIDX, detectionCenters, params )
% Perfroms spatial groupping of detections and returns a vector of IDs

spatialGroupIDs         = ones(length(currentDetectionsIDX), 1);
if useGrouping == true
    
    pairwiseDistances   = pdist2(detectionCenters, detectionCenters);
    agglomeration       = linkage(pairwiseDistances);
    % -- numSpatialGroups can calucate the identity number 
    % -- use the Detection_num / frame number(window_width) to calucate the
    % -- number of identity
    numSpatialGroups    = round(params.cluster_coeff * length(currentDetectionsIDX) / params.window_width);
    numSpatialGroups    = max(numSpatialGroups, 1);
    
    while true

         % -- use [pdist2]-> [linkage] -> [cluster] to calucate the cluster
        spatialGroupIDs     = cluster(agglomeration, 'maxclust', numSpatialGroups);
        uid = unique(spatialGroupIDs);
        % -- cal spatialGroupIDs num of each uid
        freq = [histc(spatialGroupIDs(:),uid)];
        
        largestGroupSize = max(freq);
        % The BIP solver might run out of memory for large graphs
        if largestGroupSize <= 150 
            return
        end
        
        % -- 分類還太多沒分夠，代表ID數量應該更多
        numSpatialGroups = numSpatialGroups + 1;
        
    end
    
end



