function spos = em_hog_split_pos(pos, em_sol)

% resp is examples x clusters x transformations
cluster_posteriors = sum(em_sol.resp, 3);
[~, assignments] = max(cluster_posteriors, [], 2);
num_clusters = max(assignments);

num_spos = 0;
for k = 1:num_clusters
  I = find(assignments == k);
  if length(I) >= 30
    num_spos = num_spos + 1;
    spos{num_spos} = pos(I);
  end
end
