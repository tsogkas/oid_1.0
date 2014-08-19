function [ctrs, wheel_phrase_inds] = ...
  cluster_wheel_phrases(wheel_phrases, wheels, anno, num_clusters)

conf = voc_config();

wheel_phrases_orig = wheel_phrases;
sel_inds = find(max([wheel_phrases(:).facing]) >= 0.8);
[~, dirs] = max([wheel_phrases(sel_inds).facing]);
sel_inds = sel_inds(dirs >= 3 & dirs <= 7);
wheel_phrases = wheel_phrases(sel_inds);

wheels_anno_inds = cat(2, wheels(:).anno_ind);
wheels_flipped = cat(2, wheels(:).flip);
for i = 1:length(wheel_phrases)
  % get wheels
  flip = wheel_phrases(i).flip;
  anno_ind = wheel_phrases(i).anno_ind;
  wheel_inds = anno.wheelPhrase.members{anno_ind};
  I = find(ismember(wheels_anno_inds, wheel_inds) .* (wheels_flipped == flip));

  wheel_phrase_box = wheel_phrases(i).boxes;
  wheel_boxes = cat(1, wheels(I).boxes);

  wheel_phrases(i).wheel_boxes = wheel_boxes;
  wheel_phrases(i).num_wheels = size(wheel_boxes, 1);

%  im = imreadx(wheel_phrases(i));
%  boxes = cat(1, wheel_phrase_box, wheel_boxes);
%  showboxes(im, boxes);
%
%  pause;
end

wheel_counts = cat(2, wheel_phrases(:).num_wheels);
%for i = 1:max(wheel_counts)
%  I = find(wheel_counts == i);
%  fprintf('wheel count: %d (%d)\n', i, length(I));
%  show_wheel_phrases(wheel_phrases(I(1:min(100, length(I)))));
%end

counts_to_use = [2 3];
for i = 1:length(counts_to_use)
  count = counts_to_use(i);
  I = find(wheel_counts == count);
  X = [];
  for j = 1:length(I)
    X(:,j) = wheel_phrase_feature(wheel_phrases(I(j)));
  end

  [ctrs{i}, assignments{i}] = vl_kmeans(X, num_clusters, ...
                                        'NumRepetitions', 10);

  figure(1); clf;
  show_wheel_phrase_clusters(ctrs{i});
  
  %figure(2); clf;
  for j = 1:num_clusters
    J = find(assignments{i} == j);
    fprintf('%d (%d)\n', j, length(J));
    wheel_phrase_inds{i,j} = sel_inds(I(J));
    %a = wheel_phrases(I(J));
    %b = wheel_phrases_orig(wheel_phrase_inds{i,j});
    %keyboard
    %show_wheel_phrases(wheel_phrases(I(J)));
  end
end

save([conf.paths.model_dir 'wheelPhrase_clusters_facing'], ...
     'ctrs', 'wheel_phrase_inds');



% ------------------------------------------------------------------------
function show_wheel_phrase_clusters(ctrs)
% ------------------------------------------------------------------------

num_clusters = size(ctrs, 2);
num_wheels = size(ctrs, 1)/4;
nc = 5;
nr = ceil(num_clusters/nc);
for c = 1:num_clusters
  subplot(nr, nc, c);
  boxes = reshape(ctrs(:,c), [4 num_wheels]);
  for j = 1:num_wheels
    x1 = boxes(1, j);
    x2 = boxes(3, j);
    y1 = boxes(2, j)/2;
    y2 = boxes(4, j)/2;
    line([x1 x1 x2 x2 x1]', ...
         [y1 y2 y2 y1 y1]', ...
         'color', 'b', 'linewidth', 1, 'linestyle', '-');
    axis equal;
    axis ij;
  end
end


% ------------------------------------------------------------------------
function f = wheel_phrase_feature(wheel_phrase)
% ------------------------------------------------------------------------

x1 = wheel_phrase.boxes(1);
x2 = wheel_phrase.boxes(3);
y1 = wheel_phrase.boxes(2);
y2 = wheel_phrase.boxes(4);
width = x2 - x1 + 1;
b = wheel_phrase.wheel_boxes;
b(:,[1 3]) = (b(:,[1 3]) - x1)/width;
b(:,[2 4]) = 2*(b(:,[2 4]) - y1)/width;
b = b';
f = b(:); %reshape(b, [4*size(b,2) 1]);


% ------------------------------------------------------------------------
function show_wheel_phrases(wheel_phrases)
% ------------------------------------------------------------------------

for i = 1:length(wheel_phrases)
  im = imreadx(wheel_phrases(i));
  boxes = cat(1, wheel_phrases(i).boxes, wheel_phrases(i).wheel_boxes);
  showboxes(im, boxes);
  title(sprintf('%d/%d', i, length(wheel_phrases)));
  pause;
end
