% JAR_TRAIN  Modified version of the voc5-release code to train a DPM model.
% This function trains a DPM using the OID aeroplane dataset.
%
%   model = jar_train(cls, n, root_only, align, note)
% 
% INPUT:
%   cls:         Object class (e.g., aeroplane)
%   n:           Number of aspects
%   root_only:   Train only a root model (without parts)
%   usevp:       use OID viewpoint annotations to split examples into left-
%                facing and right-facing
%   note:        Short description of the model
% 
% 

function model = jar_train(cls, n, root_only, usevp, note)

% startup;

% At every "checkpoint" in the training process we reset the 
% RNG's seed to a fixed value so that experimental results are 
% reproducible.
seed_rand();
timestamp = datestr(datevec(now()), 'dd.mmm.yyyy:HH.MM.SS');

if nargin < 3, root_only = true; end
if nargin < 4, usevp = true; end
% Default to no note
if nargin < 5
%   note = '';
  note = timestamp;
end

conf = voc_config();
oidconf = oidConfig;

% cachedir = conf.paths.model_dir;
cachedir = [oidconf.paths.modelsDirectory '/'];

diary(conf.training.log([cls '-' note]));

% Load the training data
[pos, neg, impos] = jar_data(cls, 'aeroplane', {'train','val'}, true);

% Turn facing direction vector to string and add it to pos (STATSOGK)
if usevp
    [facingStrings, ignore] = getOrientationStrings(pos);
    [pos(:).facingString] = facingStrings{:};
    pos(ignore) = []; % throw away the few unreliable left/right annotations
end

% Split foreground examples into n groups by aspect ratio
spos = split(pos, n);
lens = cellfun(@length, spos);
I = find(lens == 0);
spos(I) = [];

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
  load([cachedir cls '_lrsplit1']);
catch
  seed_rand();
  for i = 1:n
    models{i} = root_model(cls, spos{i}, note);
    % Split the i-th aspect ratio group into two clusters: 
    % left vs. right facing instances
    if usevp
        inds = splitLeftRight(spos{i}, 1);
    else
        inds = lrsplit(models{i}, spos{i});
    end
    % Train asymmetric root filter on one of these groups
    models{i} = train(models{i}, spos{i}(inds), neg_small, true, true, 1, 1, ...
                      max_num_examples, fg_overlap, 0, false, ...
                      ['lrsplit1_' num2str(i)]);
  end
  save([cachedir cls '_lrsplit1'], 'models');
end

% Train a mixture of two root filters for each aspect ratio group
% Each pair of root filters are mirror images of each other
% and correspond to two latent orientations choices
% Training uses latent positives and hard negatives
try
  load([cachedir cls '_lrsplit2']);
catch
  seed_rand();
  for i = 1:n
    % Build a mixture of two (mirrored) root filters
    models{i} = lr_root_model(models{i});
    models{i} = train(models{i}, spos{i}, neg_small, false, false, 4, 3, ...
                      max_num_examples, fg_overlap, 0, false, ...
                      ['lrsplit2_' num2str(i)]);
  end
  save([cachedir cls '_lrsplit2'], 'models');
end

% Train a mixture model composed all of aspect ratio groups and 
% latent orientation choices using latent positives and hard negatives
try 
  load([cachedir cls '_mix']);
catch
  seed_rand();
  % Combine separate mixture models into one mixture model
  model = model_merge(models);
  if root_only
    model = train(model, pos, neg_small, false, false, 6, 5, ...
                  max_num_examples, fg_overlap, num_fp, false, 'mix1');
    model = train(model, pos, neg_large, false, false, 1, 5, ...
                  max_num_examples, fg_overlap, num_fp, true, 'mix2');
  else
    model = train(model, pos, neg_small, false, false, 1, 5, ...
                  max_num_examples, fg_overlap, num_fp, false, 'mix');
  end
  save([cachedir cls '_mix'], 'model');
end

if ~root_only
  % Train a mixture model with 2x resolution parts using latent positives
  % and hard negatives
  try 
    load([cachedir cls '_parts']);
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
                  max_num_examples, fg_overlap, num_fp, false, 'parts_1');
    % Finish training by data mining on all of the negative images
    model = train(model, pos, neg_large, false, false, 1, 5, ...
                  max_num_examples, fg_overlap, num_fp, true, 'parts_2');
    save([cachedir cls '_parts'], 'model');
  end
end

% if you use sliptLeftRight instead of lrsplit in the first round of
% training to split the training examples into left- and right-facing,
% the model is 'aligned'.
if usevp, model.isAligned = true; end 
save([cachedir cls '_final'], 'model');

% test
anno = jarLoadAnno();
[ap, recall, prec] = jar_test(model, anno, 'aeroplane', 'test', true);
disp(ap);
