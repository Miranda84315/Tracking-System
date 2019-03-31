function divideTrajectories = divideTrajectories( newTrajectories )
%newTrajectories = trajectories;
inAssociation = []; tracklets = []; trackletLabels = [];
for i = 1 : length(newTrajectories)
	flag = 0;
    firstk = 0;
    lastk = 0;
	for k = 1 : length(newTrajectories(i).tracklets) 
        % case1 : is empty []
        if isempty(newTrajectories(i).tracklets(k).realdata)
            flag = 1;
            if firstk == 0
                firstk = k;
            end
        elseif flag == 1
            % case 2: is not empty, but previous is []
            [value, index] = min(newTrajectories(i).tracklets(k).realdata(:, 1));
            firstLocation = newTrajectories(i).tracklets(k).realdata(index, 3:6);
            firstLocation = feetPosition(firstLocation);
            diffLocation = firstLocation - lastLocation;
            diff = sqrt(diffLocation(1)^2 + diffLocation(2)^2)
            lastk = k;
            if diff>=800
                newTrajectories(end+1).tracklets = newTrajectories(i).tracklets(lastk:end);
                newTrajectories(end).startFrame = min([newTrajectories(end).tracklets.startFrame]);
                newTrajectories(end).endFrame = max([newTrajectories(end).tracklets.endFrame]);
                newTrajectories(end).segmentStart = min([newTrajectories(end).tracklets.segmentStart]);
                newTrajectories(end).segmentEnd = max([newTrajectories(end).tracklets.segmentEnd]);
                newTrajectories(end).feature = newTrajectories(end).tracklets(end).feature;
                
                newTrajectories(i).tracklets = newTrajectories(i).tracklets(1:firstk-1);
                break
            end
            flag = 0;
        else
            % case3 : is not empty, and previous is not []
            [value, index] = max(newTrajectories(i).tracklets(k).realdata(:, 1));
            lastLocation = newTrajectories(i).tracklets(k).realdata(index, 3:6);
            lastLocation = feetPosition(lastLocation);
        end
    end
end
divideTrajectories = newTrajectories;
end