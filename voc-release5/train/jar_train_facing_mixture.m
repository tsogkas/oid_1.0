function model = jar_train_facing_mixture(cls)
% Train a model.

% At every "checkpoint" in the training process we reset the 
% RNG's seed to a fixed value so that experimental results are 
% reproducible.
seed_rand();

conf = voc_config();
cachedir = conf.paths.model_dir;
note = '';

% Load the training data
[pos, neg] = jar_data(cls, 'aeroplane', {'train','val'}, true);

spos = jar_cluster_facing(pos);

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

try
  load([cachedir cls '_facing']);
catch
  seed_rand();
  for i = 1:length(spos)
    try
      load([cachedir cls '_facing_comp_' num2str(i)]);
      models{i} = model;
    catch
      models{i} = root_model(cls, spos{i}, note, [], [], 2);
      %[[[ Allow detections in the first level of the feature pyramid
      bl = models{i}.rules{models{i}.start}.loc.blocklabel;
      models{i}.blocks(bl).w(:) = 0;
      %]]]
      models{i} = train(models{i}, spos{i}, neg_small, true, true, 1, 1, ...
                        max_num_examples, fg_overlap, 0, false, ...
                        ['facing1_' num2str(i)]);
      models{i} = train(models{i}, spos{i}, neg_small, false, false, 4, 3, ...
                        max_num_examples, fg_overlap, 0, false, ...
                        ['facing2_' num2str(i)]);
      model = models{i};
      save([cachedir cls '_facing_comp_' num2str(i)], 'model');
    end
  end
  save([cachedir cls '_facing'], 'models');
end

% Train a mixture model composed all of aspect ratio groups and 
% latent orientation choices using latent positives and hard negatives
try 
  load([cachedir cls '_mix']);
catch
  seed_rand();
  % Combine separate mixture models into one mixture model
  model = model_merge(models);
  model = train(model, pos, neg_small, false, false, 4, 10, ...
                max_num_examples, fg_overlap, num_fp, false, 'mix_1');

  % Finish training by data mining on all of the negative images
  model = train(model, pos, neg_large, false, false, 1, 5, ...
                max_num_examples, fg_overlap, num_fp, true, 'mix_2');

  save([cachedir cls '_mix'], 'model');
end

save([cachedir cls '_final'], 'model');
