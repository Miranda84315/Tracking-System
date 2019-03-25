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
    poses = poses(find(poses(:, 5)==1), :);
    bboxes = poses(:,1:4);
    img = insertObjectAnnotation(img,'rectangle',bboxes, ones(size(bboxes,1),1));
    figure, imshow(img);
    
    bboxes = poses(find(poses(:, 5)==1),1:4);
    img = insertObjectAnnotation(img,'rectangle',bboxes, ones(size(bboxes,1),1));
    figure, imshow(img);
    close all
end
%% for openpose

img = opts.reader.getFrame(cam, frame);
poses = detections(detections(:,1) == cam & detections(:,2) == frame,3:end);

img = renderPoses(img, poses);
% Transform poses into boxes
bboxes = [];
for i = 1:size(poses,1)
    bboxes(i,:) = pose2bb(poses(i,:), opts.render_threshold);
    bboxes(i,:) = scale_bb(bboxes(i,:), poses(i,:), 1.25);
end
img = insertObjectAnnotation(img,'rectangle',bboxes, ones(size(bboxes,1),1));
figure, imshow(img);
