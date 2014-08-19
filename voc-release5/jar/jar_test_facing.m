function [ap, recall, prec] = jar_test_facing(model, anno, image_set)

scale = 0.5;

conf = voc_config();
jar_conf = jarConf();
image_ids = jarReadImageSet(anno, image_set);
image_inds = find(ismember(anno.image.id, image_ids));

parfor i = 1:length(image_inds)
  fprintf('%s: testing facing direction: %s, %d/%d\n', model.class, image_set, ...
          i, length(image_inds));

  image_ind = image_inds(i);
  image_name = anno.image.name{image_ind};
  im = imread([jar_conf.path.image image_name]);
  im = imresize(im, scale);

  [ds, bs] = imgdetect(im, model, model.thresh);

  for d = 1:8
    d_inds = find(ds(:, end-1) == d);
    if ~isempty(d_inds)
      d_ds = ds(d_inds, :);
      d_bs = bs(d_inds, :);

      d_ds = clipboxes(im, d_ds, d_bs);

      % NMS
      I = nms(d_ds, 0.5);
      d_ds = d_ds(I,:);
      ndets = size(d_ds, 1);

      % Rescale boxes
      d_ds(:, 1:4) = (d_ds(:, 1:4) - 1) / scale + 1;

      dets(d,i).box = d_ds(:, 1:4)';
      dets(d,i).score = d_ds(:, end)';
      dets(d,i).image_id = repmat(image_ids(i), [1 ndets]);
    else
      dets(d,i).box = [];
      dets(d,i).score = [];
      dets(d,i).image_id = [];
    end
  end
end

for d = 1:8
  det(d).objType = model.class;
  det(d).imageSet = image_set;
  det(d).box = cat(2, dets(d,:).box);
  det(d).score = cat(2, dets(d,:).score);
  det(d).imageId = cat(2, dets(d,:).image_id);
  det(d).attrType = 'facingDirection';
  det(d).attrLabel = d;
  [ap(d), recall{d}, prec{d}] = jarEvalDetections(det(d), anno);
end

save([conf.paths.model_dir model.class '_facing_direction_detections_' image_set], ...
     'det', 'ap', 'recall', 'prec');
