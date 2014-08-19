function spos = split_by_shape_cluster(model, pos)

conf = voc_config();
debug = false;

load([conf.paths.model_dir model.class '_shape_clusters']);
load([conf.paths.model_dir model.class '_shape_cluster_data']);

spos_ind = 0;

for c = 1:length(clusters)
  fprintf('split by cluster %d/%d\n', c, length(clusters));
  pos_inds = inds_all{c};

  for k = 1:max(clusters{c}.kIDX)
    spos_ind = spos_ind + 1;

    idx = find(clusters{c}.kIDX == k);
    % sort by distance to cluster center
    [~, ord] = sort(clusters{c}.transforms(5,idx));
    idx = idx(ord);

    rot_mu = mean(clusters{c}.transforms(4, idx));
    num_pos = 0;

    for i = 1:length(idx)
      j = pos_inds(idx(i));

      im = imreadx(pos(j));
      imsz = size(im);
      box = pos(j).boxes;
      poly = pos(j).polygon;
      rot = clusters{c}.transforms(4, idx(i)) - rot_mu;
      poly = rot_shape(poly', rot, imsz([2 1])/2)';

      imsz = size(im);
      im = imrotate(im, -rot);
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
end

end
