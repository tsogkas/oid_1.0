function em_hog_visualize_clusters(em_sol, pos)

% resp is examples x clusters x transformations
cluster_posteriors = sum(em_sol.resp, 3);
[~, assignments] = max(cluster_posteriors, [], 2);
num_clusters = max(assignments);

figure(1);
nc = 5;
nr = ceil(num_clusters/nc);
clf;
rho = em_sol.rho;
s = sqrt(size(rho,2)/32);
for j = 1:num_clusters
  fg = reshape(rho(j,:), [s s 32]);
  bg = sum(sum(fg)) / (size(fg, 1) * size(fg, 2));
  bg = repmat(bg, size(fg, 1), size(fg, 2));
  filter = log((fg .* (1-bg))./ ((1-fg) .* bg));
  vl_tightsubplot(nr, nc, j, 'margin', 0.01);
  visualizeHOG(max(0, filter));
end
drawnow;

figure(2);
for k = 1:num_clusters
  I = find(assignments == k);
  fprintf('cluster %d/%d (%d)\n', k, num_clusters, length(I));
  view_jar_data(pos(I));
%  for i = 1:length(I)
%    p = pos(I(i));
%    im = imreadx(p);
%  end
end
