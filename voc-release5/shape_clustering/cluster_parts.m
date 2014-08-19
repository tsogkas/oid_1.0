function cluster_parts(model, pos)

conf = voc_config();

load([conf.paths.model_dir model.class '_shape_cluster_data']);

numpos = length(pos);
nrules = length(model.rules{model.start});

%% parameters for shape clustering
param.DT_SIZE = [200 200];
param.SCALE = 50;
param.STEP_SIZE = 10;
param.TRANSLATION_STEPS = 1;
param.SCALES = [0.9 1 1.1];
%param.ROTATIONS = [-15 -7.5 0 7.5 15];
param.ROTATIONS = [0];
%param.ROTATIONS = [-10 0 10];
param.NUM_ITER = 20;
param.NUM_CLUSTERS = 20*ones(1, nrules);
param.INIT_ALIGNMENT = true; %false;

for c = 1:nrules
  inds = inds_all{c};
  boxes = ds_all{c};

  sh = cat(1, {pos(inds).polygon});
  sh = cellfun(@transpose, sh, 'UniformOutput', false);

  w = boxes(:,3) - boxes(:,1);
  h = boxes(:,4) - boxes(:,2);
  cx = boxes(:,1) + w/2;
  cy = boxes(:,2) + h/2;
  s = 1./w;

  for j = 1:length(sh)
    sz = size(sh{j});
    sh{j}(:,1) = (sh{j}(:,1) - repmat(cx(j), [sz(1) 1])) * s(j);
    sh{j}(:,2) = (sh{j}(:,2) - repmat(cy(j), [sz(1) 1])) * s(j);
  end

  %sh = sh(1:300);

  clusters{c} = cluster_shapes(sh, param.NUM_CLUSTERS(c), param);
  
  shapes{c} = sh;
  %view_clusters(clusters{c}, shapes{c});
  %drawnow;

  save([conf.paths.model_dir model.class '_shape_clusters'], 'clusters', 'shapes');
  %pause;
end

