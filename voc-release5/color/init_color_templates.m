function init_color_templates(model, pos, num_centers)

% Assume that model is a mixture of root filters
% for each positive example compute the latent detection
%   at the detection scale, compute the color features
%   extract the color feature subwindow and save (per component)

debug = false;
conf = voc_config();
name = model.class;
nrules = length(model.rules{model.start});

try
  load([conf.paths.model_dir name '_init_color_templates']);
catch
  numpos = length(pos);
  model.interval = conf.training.interval_fg;
  pixels = model.minsize * model.sbin / 2;
  minsize = prod(pixels);
  color_feat = cell(1,numpos);

  % compute latent filter locations and record target bounding boxes
  parfor i = 1:numpos
  %for i = 1:numpos
    par_feat{i} = cell(1,nrules);
    fprintf('%s %s: init color tpts: %d/%d\n', procid(), name, i, numpos);
    bbox = pos(i).boxes;
    % skip small examples
    if (bbox(3)-bbox(1)+1)*(bbox(4)-bbox(2)+1) < minsize
      continue;
    end
    % get example
    im = imreadx(pos(i));
    [im, bbox] = croppos(im, bbox);
    [pyra, model_dp] = gdetect_pos_prepare(im, model, bbox, 0.7);
    [ds, bs, trees] = gdetect_pos(pyra, model_dp, 1, 1, 0.7);
    if ~isempty(ds)
      % component index
      c = ds(1,end-1);
      fsz = model.filters(c).size;

      t = tree_mat_to_struct(trees{1});

      % ------------------------
      % Temp code to make features 2x resolution
      % ------------------------
      if t(1).l <= model.interval
        continue;
      end
      t(1).x = (t(1).x-1)*2+1;
      t(1).y = (t(1).y-1)*2+1;
      t(1).l = t(1).l - model.interval;
      t(1).ds = t(1).ds + 1;
      fsz = fsz*2;
      % ------------------------

      f = extract_color_feat(pyra, fsz, t(1).ds, ...
                             t(1).x, t(1).y, t(1).l);

      % stash index of positive example that this came from
      % at the end of the feature array (ugly, yes)
      par_feat{i}{c} = cat(2, par_feat{i}{c}, [f(:); i]);

      if debug
        figure(1);
        visualize_color_feat(f, model.features.color_ctrs);
        figure(2);
        showboxes(im, ds);
        pause;
      end
    end
  end
  color_feat_all = [];
  comp_feat = cat(1, par_feat{:});
  for i = 1:nrules
    color_feat_all{i} = cat(2, comp_feat{:,i});
  end

  save([conf.paths.model_dir name '_init_color_templates'], ...
       'color_feat_all');
end

for i = 1:nrules
  fsz = model.filters(i).size;
  % ------------------------
  % Temp code to make features 2x resolution
  % ------------------------
  fsz = fsz*2;
  % ------------------------

  % features without pos index
  X = color_feat_all{i}(1:end-1, :);
  N = size(X, 2);

  % Exponentiate and renormalize to make
  % histograms peaky
  X = reshape(X, [fsz(1) fsz(2) 30 N]);
  X = X.^12;
  X = X ./ repmat(sum(X, 3), [1 1 30 1]);
  X = reshape(X, [fsz(1)*fsz(2)*30 N]);

  % pos array index
  pos_inds = color_feat_all{i}(end, :);
  [C, A] = vl_kmeans(X, num_centers, 'verbose', ...
                     'NumRepetitions', 10);

  figure(1); clf;
  ims = cell(1, num_centers);
  for j = 1:num_centers
    cluster_inds = find(A == j);
    f = reshape(C(:,j), [fsz size(model.features.color_ctrs,2)]);
    ims{j} = visualize_color_feat(f, model.features.color_ctrs);
  end
  montage(cat(4, ims{:}));

  %figure(1); clf;
  figure(2); clf;
  for j = 1:num_centers
    cluster_inds = find(A == j);
    fprintf('Component %d (%d)\n', j, length(cluster_inds));
    figure(2);
    subplot(2,1,1);
    title(num2str(j));
    pos_montage(pos(pos_inds(cluster_inds)), 80);

    f = X(:, cluster_inds);
    color_feat_montage(f, fsz, model.features.color_ctrs);
    
    %f = reshape(C(:,j), [fsz size(model.features.color_ctrs,2)]);
    %figure(1);
    %subplot(num_centers,1,j);
    %visualize_color_feat(f, model.features.color_ctrs);
    %axis off;
    pause;
  end
end


% ------------------------------------------------------------------------
function color_feat_montage(f, fsz, ctrs)

sz = [fsz size(ctrs, 2)];
ims = zeros([fsz 3 size(f, 2)], 'uint8');
for i = 1:size(f, 2)
  ims(:,:,:,i) = color_feat_to_im(reshape(f(:, i), sz), ctrs);
end
figure(2);
subplot(2,1,2);
montage(ims, 'size', [nan round(sqrt(size(ims,4)))]);
% ------------------------------------------------------------------------


% ------------------------------------------------------------------------
function f = extract_color_feat(pyra, fsz, ds, x, y, l)
% ------------------------------------------------------------------------

% remove virtual padding
fy = y - pyra.pady*(2^ds-1);
fx = x - pyra.padx*(2^ds-1);
f = pyra.color_feat{l}(fy:fy+fsz(1)-1, fx:fx+fsz(2)-1, :);
