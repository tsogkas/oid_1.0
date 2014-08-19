function clusters = EM_shapes(pos)

% X(i, n, t)  feature i, example index, transformation
% X(:, n, t)  features for example n, transformation t
% X(:, :, t)  features for examples 1:N, transformation t

% X{n, t} = sparse linear indices of non-zero elements

conf = voc_config();

%% parameters for shape clustering
param.TEMPLATE_SIZE = [100 100];
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

for j = 1:length(shapes)
  shape = shapes{j};
  mu = repmat(mean(shape), [size(shape,1) 1]);
  %sc = sqrt((max(shape(:,1)) - min(shape(:,1)))^2 + ...
  %          (max(shape(:,2)) - min(shape(:,2)))^2);
  sc = max(shape(:,2)) - min(shape(:,2));
  shape(:,1) = (shape(:,1) - mu(:,1)) / sc;
  shape(:,2) = (shape(:,2) - mu(:,2)) / sc;
  shapes{j} = shape;
end

[X, transforms] = make_xformed_data(shapes, param);
save_fn = @(best) save_fn_base(best, pos(1).obj_type, shapes, transforms, conf, param);
best = EM_Bernoulli_xform(X, param.NUM_CLUSTERS, 1, param.NUM_ITER, save_fn);
clusters.EM = best;
clusters.transforms = transforms;
clusters.param = param;

save([conf.paths.model_dir pos(1).obj_type '_EM_shape_clusters_facing'], 'clusters', 'shapes', 'sel_inds');

try
  figure(param.NUM_ITER+1);
  EM_view_clusters(clusters, shapes);
  drawnow;
catch
end


% ------------------------------------------------------------------------
function save_fn_base(best, obj_type, shapes, transforms, conf, param)
% ------------------------------------------------------------------------
clusters.EM = best;
clusters.transforms = transforms;
clusters.param = param;
save([conf.paths.model_dir obj_type '_EM_shape_clusters_facing'], 'clusters', 'shapes');



% ------------------------------------------------------------------------
function [X, transforms] = make_xformed_data(shapes, param)
% ------------------------------------------------------------------------

TPT_SIZE = param.TEMPLATE_SIZE;
TPT_CENTER = round(TPT_SIZE([2 1])/2);
SCALE = param.SCALE;

% initialize the initial search space
stepsize = param.STEP_SIZE;
n = param.TRANSLATION_STEPS;
[offx, offy] = meshgrid(-n:n,-n:n);
offx = offx(:)*stepsize; offy = offy(:)*stepsize;
offs = param.SCALES;

num_xforms = 1;

X = cell(length(shapes), num_xforms);

%% Estimate initial transforms by aligning it to the mean
init_transforms = zeros(4, length(shapes));
fprintf('Estimating initial tranforms [%i total]\n', length(shapes));
fprintf('Search space : offx [%.1f %.1f] step:%.1f, offy [%.1f %.1f] step:%.1f, offs [%.1f %.1f]\n', ...
        min(offx), max(offx), stepsize, min(offy),max(offy), stepsize, min(offs), max(offs));

%for i = 1:length(shapes)
%  if mod(i,100) == 0
%    fprintf('%i..', i);
%  end
%  shape = shapes{i};
%  cx = mean(shape, 1);
%
%  pts = (shape - repmat(cx, size(shape,1), 1)) * SCALE + ...
%        repmat(TPT_CENTER, size(shape,1), 1);
%
%  %pts = shape * SCALE + ...
%  %      repmat(TPT_CENTER, size(shape,1), 1);
%
%  boundary = poly2boundary(pts);
%  out_of_image = boundary(:,1) < 1 | boundary(:,1) > TPT_SIZE(2) | ...
%                 boundary(:,2) < 1 | boundary(:,2) > TPT_SIZE(1);
%  boundary = boundary(~out_of_image,:);
%  boundary_inds = sub2ind(TPT_SIZE, boundary(:,2), boundary(:,1));
%
%  X{i,1} = boundary_inds;
%end

t = 1;

for ds = [0.9 1 1.1]
  for dx = -10:10:10
    for dy = -10:10:10
      %if ds == 0 && dx == 0 && dy == 0, continue, end;

      for i = 1:length(shapes)
        if mod(i,100) == 0
          fprintf('%i..', i);
        end
        shape = shapes{i};
        cx = mean(shape, 1);

        pts = (shape - repmat(cx, size(shape,1), 1)) * ds * SCALE + ...
              repmat(TPT_CENTER + [dx dy], size(shape,1), 1);

        boundary = poly2boundary(pts);
        out_of_image = boundary(:,1) < 1 | boundary(:,1) > TPT_SIZE(2) | ...
                       boundary(:,2) < 1 | boundary(:,2) > TPT_SIZE(1);
        boundary = boundary(~out_of_image,:);
        boundary_inds = sub2ind(TPT_SIZE, boundary(:,2), boundary(:,1));

        X{i,t} = boundary_inds;
      end

      transforms(t).scale = ds;
      transforms(t).trans = [dx dy];
      transforms(t).rot = 0;

      t = t + 1;
    end
  end
end

%for i = 1:length(shapes)
%  if mod(i,100) == 0
%    fprintf('%i..', i);
%  end
%  shape = shapes{i};
%  cx = mean(shape, 1);
%
%  rad = (-30)/180*pi;
%  R = [cos(rad) -sin(rad); ...
%       sin(rad)  cos(rad)];
%
%  pts = (shape - repmat(cx, size(shape,1), 1)) * SCALE;
%  pts = (R*pts')';
%  pts = pts + repmat(TPT_CENTER, size(shape,1), 1);
%
%  %pts = shape * SCALE + ...
%  %      repmat(TPT_CENTER, size(shape,1), 1);
%
%  boundary = poly2boundary(pts);
%  out_of_image = boundary(:,1) < 1 | boundary(:,1) > TPT_SIZE(2) | ...
%                 boundary(:,2) < 1 | boundary(:,2) > TPT_SIZE(1);
%  boundary = boundary(~out_of_image,:);
%  boundary_inds = sub2ind(TPT_SIZE, boundary(:,2), boundary(:,1));
%
%  X{i,2} = boundary_inds;
%end

fprintf('[done]\n');
