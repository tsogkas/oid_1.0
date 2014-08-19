function EM_view_clusters(clusters, shapes)

num_shapes = length(shapes);

k_ind = zeros(1, num_shapes);
t_ind = zeros(1, num_shapes);
for i = 1:num_shapes
  gamma = clusters.EM.resp(i, :, :);
  gamma = reshape(gamma, size(gamma, 2), size(gamma, 3));
  [~, ind] = max(gamma(:));
  [t, k] = ind2sub(size(gamma), ind);
  k_ind(i) = k;
  t_ind(i) = t;
end

NUM_CLUSTERS = max(k_ind);
% draw the clusters
nx = 5;
ny = ceil(NUM_CLUSTERS/nx);
for i = 1:NUM_CLUSTERS
  idx = find(k_ind == i);
  subplot(ny,nx,i); hold on; axis ij; 
  for j = 1:length(idx)
    shape = shapes{idx(j)};
    normshape = xform_shape(shape, clusters.transforms(t_ind(idx(j))), clusters.param);
    plot(normshape([1:end 1], 1), normshape([1:end 1],2),'-','Color',rand(3,1));
  end
  axis tight;       
  axis off;
  axis equal;
  title(sprintf('%i', length(idx)));
end


% ------------------------------------------------------------------------
function shape = xform_shape(shape, xform, param)
% ------------------------------------------------------------------------
mu = mean(shape, 1);
shape = (shape - repmat(mu, size(shape, 1), 1)) * xform.scale * param.SCALE + ...
        repmat(xform.trans, size(shape, 1), 1);


