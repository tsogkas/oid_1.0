function view_cluster_examples(clusters, model, pos)

conf = voc_config();

load([conf.paths.model_dir model.class '_shape_cluster_data']);

numpos = length(pos);
nrules = length(model.rules{model.start});

if isfield(clusters{1}, 'EM')
  for c = 1:nrules
    num_shapes = size(clusters{c}.EM.resp,1);
    k_ind = zeros(1, num_shapes);
    for i = 1:num_shapes
      gamma = clusters{c}.EM.resp(i, :, :);
      gamma = reshape(gamma, size(gamma, 2), size(gamma, 3));
      [~, ind] = max(gamma(:));
      [t, k] = ind2sub(size(gamma), ind);
      k_ind(i) = k;
    end
    clusters{c}.kIDX = k_ind;
  end
end


for c = 1:nrules
  inds = inds_all{c};

  if isfield(clusters{c}, 'kDT')
    % draw the average distance transforms
    kDT = reshape(clusters{c}.kDT, size(clusters{c}.kDT,1), ...
                  size(clusters{c}.kDT,2), 1, size(clusters{c}.kDT,3));
    figure(1); clf;
    maxval = max(kDT(:));
    montage(kDT/maxval); colormap hot;
    title('Cluster Mean DT');
  end

  for k = 1:max(clusters{c}.kIDX)
    if isfield(clusters{c}, 'EM')
      figure(1); clf;
      imagesc(reshape(clusters{c}.EM.rho{k}, 100, 100))
    end

    idx = find(clusters{c}.kIDX == k);
    % sort by distance to cluster center
    [~, ord] = sort(clusters{c}.transforms(5,idx));
    idx = idx(ord);

    rot_mu = mean(clusters{c}.transforms(4, idx));
    disp(rot_mu);

    for i = 1:min(30, length(idx))
      j = inds(idx(i));

      im = imreadx(pos(j));
      imsz = size(im);
      box = pos(j).boxes;
      poly = pos(j).polygon;
      %poly(1,:) = poly(1,:) - box(1) + 1;
      %poly(2,:) = poly(2,:) - box(2) + 1;
      rot = clusters{c}.transforms(4, idx(i)) - rot_mu;
      poly = rot_shape(poly', rot, imsz([2 1])/2)';

      %box = round(box);
      %im = im(box(2):box(4), box(1):box(3), :);
      imsz = size(im);
      im = imrotate(im, -rot);
      imshift = (size(im) - imsz)/2;
      poly(1,:) = poly(1,:) + imshift(2);
      poly(2,:) = poly(2,:) + imshift(1);
      box = [min(poly(1,:)) min(poly(2,:)) max(poly(1,:)) max(poly(2,:))];

      figure(2); clf;
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
