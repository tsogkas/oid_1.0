function model = jar_train_from_shape_clusters(cls, root_only, note)
% Train a model.
%
% cls     Object class (e.g., aeroplane)
% n       Number of aspects

% At every "checkpoint" in the training process we reset the 
% RNG's seed to a fixed value so that experimental results are 
% reproducible.
seed_rand();

% Default to no note
if nargin < 3
  note = '';
end

conf = voc_config();
cachedir = conf.paths.model_dir;
%swamp_limit = 30;

% Load the training data
[pos, neg] = jar_data(cls, 'aeroplane', {'train','val'}, true);

% Split foreground examples into n groups by aspect ratio
try
  load([cachedir cls '_shape_cluster_facing_pos']);
catch  
  cluster_parts_facing(pos);
  spos = split_by_shape_cluster_facing(pos);
  save([cachedir cls '_shape_cluster_facing_pos'], 'spos');
end
%pos = cat(2, spos{:});
n = length(spos);
%for i = 1:n
%  spos_lim{i} = spos{i}(1:min(swamp_limit, length(spos{i})));
%end

max_num_examples = conf.training.cache_example_limit;
num_fp           = conf.training.wlssvm_M;
fg_overlap       = conf.training.fg_overlap;

% Select a small, random subset of negative images
% All data mining iterations use this subset, except in a final
% round of data mining where the model is exposed to all negative
% images
num_neg   = length(neg);
neg_perm  = neg(randperm(num_neg));
neg_small = neg_perm(1:min(num_neg, conf.training.num_negatives_small));
neg_large = neg; % use all of the negative images

% Train one asymmetric root filter for each aspect ratio group
% using warped positives and random negatives
try
  load([cachedir cls '_shape_init1']);
catch
  seed_rand();
  for i = 1:n
    models{i} = root_model(cls, spos{i}, note);
    % Train asymmetric root filter on one of these groups
    models{i} = train(models{i}, spos{i}, neg_small, true, true, 1, 1, ...
                      max_num_examples, fg_overlap, 0, false, ...
                      ['shape_init1_' num2str(i)]);
  end
  save([cachedir cls '_shape_init1'], 'models');
end

% Train a mixture of two root filters for each aspect ratio group
% Each pair of root filters are mirror images of each other
% and correspond to two latent orientations choices
% Training uses latent positives and hard negatives
try
  load([cachedir cls '_shape_init2']);
catch
  seed_rand();
  for i = 1:n
    models{i} = train(models{i}, spos{i}, neg_small, false, false, 4, 3, ...
                      max_num_examples, fg_overlap, 0, false, ...
                      ['shape_init2_' num2str(i)]);
  end
  save([cachedir cls '_shape_init2'], 'models');
end



% Train a mixture model composed all of aspect ratio groups and 
% latent orientation choices using latent positives and hard negatives
try 
  load([cachedir cls '_shape_mix']);
catch
  seed_rand();
  % Combine separate mixture models into one mixture model
  for i = 1:n
    models{i} = lr_root_model(models{i});
  end
  model = model_merge(models);

  if root_only
    model = train(model, pos, neg_small, false, false, 6, 5, ...
                  max_num_examples, fg_overlap, num_fp, false, 'shape_mix1');
    model = train(model, pos, neg_large, false, false, 1, 5, ...
                  max_num_examples, fg_overlap, num_fp, true, 'shape_mix2');
  else
    model = train(model, pos, neg_small, false, false, 1, 5, ...
                  max_num_examples, fg_overlap, num_fp, false, 'shape_mix');
  end
  save([cachedir cls '_shape_mix'], 'model');
end

if ~root_only
  % Train a mixture model with 2x resolution parts using latent positives
  % and hard negatives
  try 
    load([cachedir cls '_shape_parts']);
  catch
    seed_rand();
    % Add parts to each mixture component
    for i = 1:2:2*n
      % Top-level rule for this component
      ruleind = i;
      % Top-level rule for this component's mirror image
      partner = i+1;
      % Filter to interoplate parts from
      filterind = i;
      model = model_add_parts(model, model.start, ruleind, ...
                              partner, filterind, 8, [6 6], 1);
      % Enable learning location/scale prior
      bl = model.rules{model.start}(i).loc.blocklabel;
      model.blocks(bl).w(:)     = 0;
      model.blocks(bl).learn    = 1;
      model.blocks(bl).reg_mult = 1;
    end
    % Train using several rounds of positive latent relabeling
    % and data mining on the small set of negative images
    model = train(model, pos, neg_small, false, false, 8, 10, ...
                  max_num_examples, fg_overlap, num_fp, false, 'shape_parts_1');
    % Finish training by data mining on all of the negative images
    model = train(model, pos, neg_large, false, false, 1, 5, ...
                  max_num_examples, fg_overlap, num_fp, true, 'shape_parts_2');
    save([cachedir cls '_shape_parts'], 'model');
  end
end

save([cachedir cls '_shape_final'], 'model');

% test
anno = jarLoadAnno();
[ap, recall, prec] = jar_test(model, anno, 'aeroplane', 'test', true, '_shape');
disp(ap);
