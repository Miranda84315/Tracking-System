% Demo visualizing OpenPose detections
opts = get_opts();

cam = 1;

load(fullfile(opts.detection,'top1', sprintf('camera%d.mat',cam)));
ind = dlmread(fullfile(opts.detection,'top1', sprintf('index%d.txt',cam)));
ind = ind + 1;
detections = detections(ind, :);


%% 
for frame = 122900:123030
    % disappear -> 122911
    % start -> 123029
    img = opts.reader.getFrame(cam, frame);
    poses = detections(detections(:,1) == cam & detections(:,2) == frame,3:end);

    % Transform poses into boxes
    bboxes = poses(:,1:4);
    img = insertObjectAnnotation(img,'rectangle',bboxes, ones(size(bboxes,1),1));
    figure, imshow(img);
    close all
end

