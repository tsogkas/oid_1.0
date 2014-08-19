function view_detections(det, anno, do_sorted)

jar_conf = jarConf();

image_ids = unique(det.imageId);

if do_sorted
  %[[[ Show detections sorted by confidence
  [~, ord] = sort(det.score, 'descend');
  det.box = det.box(:,ord);
  det.score = det.score(ord);
  det.imageId = det.imageId(ord);
  if isfield(det, 'filterBoxes')
    det.filterBoxes = det.filterBoxes(:,ord);
  end
  for i = 1:length(det.score)
    image_id = det.imageId(i);

    image_name = anno.image.name{find(anno.image.id == image_id)};
    im = imread([jar_conf.path.image image_name]);

    if isfield(det, 'filterBoxes')
      showboxes(im, det.filterBoxes(:,i)');
    else
      showboxes(im, det.box(:,i)');
    end
    title(sprintf('%d score %.3f', i, det.score(i)));
    pause;
  end
  %]]]
else
  %[[[ Show each detection in an image
  for i = 1:length(image_ids)
    image_id = image_ids(i);

    image_name = anno.image.name{find(anno.image.id == image_id)};
    im = imread([jar_conf.path.image image_name]);

    inds = find(det.imageId == image_id);
    [~, ord] = sort(det.score(inds), 'descend');

    for j = inds(ord)
      showboxes(im, det.box(:,j)');
      title(sprintf('score %.3f', det.score(j)));
      pause;
    end
  end
  %]]]
end
