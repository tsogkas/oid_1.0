function jobs = start_jobs(fn, k, note, cores)

objs = {'nose', 'verticalStabilizer', 'wingPhrase', 'wheelPhrase'};

jobs = cell(length(objs), 1);

for i = 1:length(jobs)
  fprintf('\nstarting %s\n\n', objs{i});
  jobs{i} = batch(fn, 0, {objs{i}, k, true, note}, ...
                  'matlabpool', cores, 'CaptureDiary', false);
end
