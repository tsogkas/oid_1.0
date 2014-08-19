function [best, sel_inds] = em_hog_part_shape_and_im(pos, num_clusters)

conf = voc_config();

sel_inds = find(max([pos(:).facing]) >= 0.8);
[~, dirs] = max([pos(sel_inds).facing]);
sel_inds = sel_inds(dirs >= 3 & dirs <= 7);

try
  load([conf.paths.model_dir pos(1).obj_type '_em_hog_data_shape_and_im_facing']);
catch
  %anno = jarLoadAnno();
  %[X, num_xforms] = em_hog_data_shape_and_im(pos(sel_inds), anno);
  [X, num_xforms] = em_hog_data_shape_and_im(pos(sel_inds));
  save([conf.paths.model_dir pos(1).obj_type '_em_hog_data_shape_and_im_facing'], ...
       'X', 'num_xforms');
end

best = em_hog(X, num_xforms, num_clusters, 5, 10);

save([conf.paths.model_dir pos(1).obj_type '_em_hog_shape_and_im_clusters_facing'], ...
      'best', 'sel_inds');


