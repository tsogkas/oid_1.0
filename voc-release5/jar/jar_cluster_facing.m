function spos = jar_cluster_facing(pos, anno)

X = cat(2, pos(:).facing);
K = size(X, 1);
[~, argmax] = max(X);
for k = 1:K
  inds = find(argmax == k);
  fprintf('Cluster %d (%d)\n', k, length(inds));
  spos{k} = pos(inds);
  N = min(length(inds), 100);
  %view_jar_data(pos(inds(1:N)));
end

%K = size(anno.aeroplane.facingDirection, 1);
%[~, argmax] = max(anno.aeroplane.facingDirection);
%for k = 1:K
%  I = find(argmax == k);
%  fprintf('Cluster %d (%d)\n', k, length(I));
%  inds = find(ismember([pos(:).anno_ind], I));
%  spos{k} = pos(inds);
%  %view_jar_data(pos(inds));
%end

%X = anno.aeroplane.facingDirection';
%K = size(X, 2);
%[idx, C] = kmeans(X, K, 'replicates', 5, 'emptyaction', 'drop');
%
%for k = 1:K
%  I = find(idx == k);
%  inds = find(ismember([pos(:).anno_ind], I));
%  view_jar_data(pos(inds));
%end
