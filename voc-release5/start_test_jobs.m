function jobs = start_test_jobs(tag, cores)
% tag should be _em_hog

conf = voc_config();
anno = jarLoadAnno();

function [ap, recall, prec] = jar_test(model, anno, obj_class, ...
                                       image_set, only_typical, infix)


%objs = {'nose', 'verticalStabilizer', 'wingPhrase', 'wheelPhrase'};
objs = {'nose'};

jobs = cell(length(objs), 1);
jobs2 = cell(length(objs), 1);

for i = 1:length(jobs)
  load([conf.paths.model_dir objs{i} tag '_final']);
  fprintf('\nstarting %s\n\n', objs{i});

  jobs{i}  = batch(@jar_test, 0, {model, anno, objs{i}, {'train','val'}, false, tag}, 'matlabpool', cores, 'CaptureDiary', false);
  jobs2{i} = batch(@jar_test, 0, {model, anno, objs{i}, {'test'       }, false, tag}, 'matlabpool', cores, 'CaptureDiary', false);
end
