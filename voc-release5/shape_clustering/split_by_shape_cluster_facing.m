function spos = split_by_shape_cluster_facing(pos)

conf = voc_config();
MIN_CLUSTER_SIZE = 30;
debug = false;

cls = pos(1).obj_type;
load([conf.paths.model_dir cls '_shape_clusters_facing']);

spos_ind = 0;
num_clusters = max(clusters.kIDX);

for k = 1:num_clusters
  fprintf('..[%d/%d]', k, num_clusters);
  idx = find(clusters.kIDX == k);
  
  % Skip small clusters
  if length(idx) < MIN_CLUSTER_SIZE, continue, end;

  spos_ind = spos_ind + 1;

  % sort by distance to cluster center
  [~, ord] = sort(clusters.transforms(5,idx));
  idx = idx(ord);

  rot_mu = mean(clusters.transforms(4, idx));
  num_pos = 0;

  for i = 1:length(idx)
    tic_toc_print('..%d', num_pos);
    j = sel_inds(idx(i));

    im = imreadx(pos(j));
    imsz = size(im);
    box = pos(j).boxes;
    poly = pos(j).polygon;
    rot = clusters.transforms(4, idx(i)) - rot_mu;
    poly = rot_shape(poly', rot, imsz([2 1])/2)';

    imsz = size(im);
    if rot ~= 0
      im = imrotate(im, -rot);
    end
    imshift = (size(im) - imsz)/2;
    poly(1,:) = poly(1,:) + imshift(2);
    poly(2,:) = poly(2,:) + imshift(1);
    box = [min(poly(1,:)) min(poly(2,:)) max(poly(1,:)) max(poly(2,:))];

    num_pos = num_pos + 1;
    p = pos(j);
    p.boxes = box;
    p.polygon = poly;
    p.im_rotate = -rot;
    spos{spos_ind}(num_pos) = p;

    if debug
      figure(1); clf;
      im = imreadx(p);
      [im, box, poly] = croppos(im, p.boxes, p.polygon);
      %showboxes(im, pos(j).boxes);
      showboxes(im, box);
      %imagesc(im); axis image; axis off;
      hold on;
      plot([poly(1,:) poly(1,1)], ...
           [poly(2,:) poly(2,1)], ...
           '-', 'Color', 'r', 'LineWidth', 2);
      hold off;
      title(sprintf('cluster: %d  %d/%d', k, i, length(idx)));
      %fprintf(sprintf('%s\n', pos(i).im));
      pause;
    end
  end
end
fprintf('..[done]\n');
