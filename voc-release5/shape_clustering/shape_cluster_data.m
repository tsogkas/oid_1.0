function shape_cluster_data(model, pos)

conf = voc_config();

try
  load([conf.paths.model_dir model.class '_shape_cluster_data']);
catch
  numpos = length(pos);
  model.interval = conf.training.interval_fg;
  pixels = model.minsize * model.sbin / 2;
  minsize = prod(pixels);
  nrules = length(model.rules{model.start});
  pard = cell(1,numpos);
  pari = cell(1,numpos);

  % compute latent filter locations and record target bounding boxes
  parfor i = 1:numpos
    pard{i} = cell(1,nrules);
    pari{i} = cell(1,nrules);
    fprintf('%s %s: shape cluster data: %d/%d\n', procid(), model.class, i, numpos);
    bbox = pos(i).boxes;
    % skip small examples
    if (bbox(3)-bbox(1)+1)*(bbox(4)-bbox(2)+1) < minsize
      continue;
    end
    % get example
    im = imreadx(pos(i));
    [im, bbox] = croppos(im, bbox);
    [pyra, model_dp] = gdetect_pos_prepare(im, model, bbox, 0.7);
    [ds, bs] = gdetect_pos(pyra, model_dp, 1, ...
                            1, 0.7, [], 0.5);
    if ~isempty(ds)
      % component index
      c = ds(1,end-1);
      pard{i}{c} = [pard{i}{c}; ds(:,1:end-2)];
      pari{i}{c} = [pari{i}{c}; i];
    end
  end
  ds_all = cell(1,nrules);
  inds_all = cell(1,nrules);
  for i = 1:numpos
    for c = 1:nrules
      ds_all{c} = [ds_all{c}; pard{i}{c}];
      inds_all{c} = [inds_all{c}; pari{i}{c}];
    end
  end
  save([conf.paths.model_dir model.class '_shape_cluster_data'], 'ds_all', 'inds_all');
end
