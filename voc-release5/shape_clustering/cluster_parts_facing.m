function cluster_parts_facing(pos)

conf = voc_config();

%% parameters for shape clustering
param.DT_SIZE = [200 200];
param.SCALE = 50;
param.STEP_SIZE = 10;
param.TRANSLATION_STEPS = 1;
param.SCALES = [0.9 1 1.1];
%param.ROTATIONS = [-15 -7.5 0 7.5 15];
%param.ROTATIONS = [-7.5 0 7.5];
param.ROTATIONS = [0];
param.NUM_ITER = 10;
param.NUM_CLUSTERS = 20;
param.INIT_ALIGNMENT = true;

sel_inds = find(max([pos(:).facing]) >= 0.8);
[~, dirs] = max([pos(sel_inds).facing]);
sel_inds = sel_inds(dirs >= 3 & dirs <= 7);

shapes = cat(1, {pos(sel_inds).polygon});
shapes = cellfun(@transpose, shapes, 'UniformOutput', false);

clusters = cluster_shapes(shapes, param.NUM_CLUSTERS, param);
save([conf.paths.model_dir pos(1).obj_type '_shape_clusters_facing'], ...
      'clusters', 'shapes', 'sel_inds');

try
  view_clusters(clusters, shapes);
  drawnow;
catch
end
