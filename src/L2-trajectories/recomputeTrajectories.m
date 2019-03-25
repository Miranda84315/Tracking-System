function newTrajectories = recomputeTrajectories( newTrajectories )
%RECOMPUTETRAJECTORIES Summary of this function goes here
%   Detailed explanation goes here

segmentLength = 50;

for i = 1:length(newTrajectories)

    segmentStart = newTrajectories(i).segmentStart;
    segmentEnd = newTrajectories(i).segmentEnd;
    
    numSegments = (segmentEnd + 1 - segmentStart) / segmentLength;
    
    alldata = {newTrajectories(i).tracklets(:).data};
    alldata = cell2mat(alldata');
    alldata = sortrows(alldata,2);
    [~, uniqueRows] = unique(alldata(:,1));
    
    alldata = alldata(uniqueRows,:);
    dataFrames = alldata(:,1);
    
    frames = segmentStart:segmentEnd;
    interestingFrames = round([min(dataFrames), frames(1) + segmentLength/2:segmentLength:frames(end),  max(dataFrames)]);
        
    keyData = alldata(ismember(dataFrames,interestingFrames),:);
    
%     for k = size(keyData,1)-1:-1:1
%         
%         while keyData(k,2) == keyData(k+1,2)
%             keyData(k+1,:) = [];
%         end
%         
%     end
    
    keyData(:,2) = -1;
    newData = fillTrajectories(keyData);
    
    newTrajectory = newTrajectories(i);
    sampleTracklet = newTrajectories(i).tracklets(1);
    newTrajectory.tracklets = [];
    
    realdata = {newTrajectories(i).tracklets(:).realdata};
    realdata = cell2mat(realdata');
    
    originalFeature = {newTrajectories(i).tracklets(:).feature};
    true_length = [newTrajectories(i).tracklets.segmentStart];
    
    temp=1;
    for k = 1:numSegments
       
        tracklet = sampleTracklet;

        tracklet.segmentStart = segmentStart + (k-1)*segmentLength;
        tracklet.segmentEnd   = tracklet.segmentStart + segmentLength - 1;
        
        trackletFrames = tracklet.segmentStart:tracklet.segmentEnd;
        if ismember(tracklet.segmentStart, true_length )
            tracklet.feature = cell2mat(originalFeature(temp));
            temp = temp+1;
        else
            tracklet.feature = [];
        end
        
        rows = ismember(newData(:,1), trackletFrames);
        rows2 = ismember(realdata(:,1), trackletFrames);
        
        tracklet.realdata =  realdata(rows2,:);
        tracklet.center =  [median(newData(rows,3)), median(newData(rows,4))] ;

        tracklet.data = newData(rows,:);
        
        tracklet.startFrame = min(tracklet.data(:,1));
        tracklet.endFrame = max(tracklet.data(:,1));
         if isempty(tracklet.data)
             tracklet.startFrame = min(tracklet.realdata(:,1));
             tracklet.endFrame = max(tracklet.realdata(:,1));
         end
        newTrajectory.startFrame = min(newTrajectory.startFrame, tracklet.startFrame);
        newTrajectory.endFrame = max(newTrajectory.endFrame, tracklet.endFrame);
        
        if ~isempty(tracklet.data)
            newTrajectory.tracklets = [newTrajectory.tracklets; tracklet];
        end
        
    end
    
    newTrajectories(i) = newTrajectory;
    

end

