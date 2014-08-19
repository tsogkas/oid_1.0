function ap = jar_test_mean_box(only_typical)

conf = voc_config();
jar_conf = jarConf();
anno = jarLoadAnno();

%% Get training boxes
pos = jar_data('aeroplane', 'aeroplane', {'train','val'}, only_typical);
boxes = cat(1, pos(:).boxes)';

%% Mean box in normalized image coordinates
aero_inds = cat(2, pos(:).anno_ind);
image_ids = anno.aeroplane.imageId(aero_inds);
image_inds = vl_binsearch(anno.image.id, image_ids);
sizes = anno.image.size(:, image_inds);
boxes = (2*(boxes-1)+1) ./ repmat(sizes([2 1],:), [2 1]);
mbox = mean(boxes, 2);

%% Read test image set
[image_ids, tag] = jarReadImageSet(anno, 'aeroplane', 'test', only_typical);
image_inds = find(ismember(anno.image.id, image_ids));
num_images = length(image_ids);

%% "Predict" test boxes
pred_boxes = repmat(mbox, [1 num_images]);
pred_boxes = pred_boxes .* repmat(anno.image.size([2 1], image_inds), [2 1]);

%% Test
det.objType = 'aeroplane';
det.imageSet = 'test';
det.box = pred_boxes;
det.imageId = image_ids;

ap = zeros(1, T);
for i = 1:T
  det.score = rand(1, num_images);
  ap(i) = jarEvalDetections(det, anno);
  fprintf('AP %d: %.3f\n', i, ap(i));
end

view_detections(det, anno, true);
