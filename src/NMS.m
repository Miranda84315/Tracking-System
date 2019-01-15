function detections = NMS(opts, detections_original, iCam)
command = strcat('C:\Users\Owner\Anaconda3\envs\tensorflow\python.exe src/nms.py' , ...
    sprintf(' --icam %s', iCam));
command
system(command);

ind = dlmread(fullfile(opts.detection,'top1', sprintf('index%d.txt',iCam)));
ind = ind + 1;
detections = detections_original(ind, :);

end