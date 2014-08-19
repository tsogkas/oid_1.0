% OIDCONFIG
% Configure paths, parameter values and settings for an OID project.
% 
% Depending on what you want to do, you must set the parameter values
% accordingly. Below are some instructions:
% 
% --- Collect positive examples scores to calculate empirical thresholds
% conf.threshPerNode = 0;
% conf.storeFullScores = 1;
% 
% --- Test using empirical thresholds
% conf.threshPerNode   = 1;
% conf.storeFullScores = *;


function conf = oidConfig()

% paths for data, models, cascade files, results etc.
conf.paths.rootDirectory    = pwd;
conf.paths.dataDirectory    = fullfile(conf.paths.rootDirectory,'data');
conf.paths.modelsDirectory  = fullfile(conf.paths.dataDirectory,'models');
conf.paths.cascadeDirectory = fullfile(conf.paths.dataDirectory,'cascade');
conf.paths.resultsDirectory = fullfile(conf.paths.dataDirectory,'results');
conf.paths.setsDirectory    = fullfile(conf.paths.dataDirectory,'sets');
conf.paths.imageDirectory   = fullfile(conf.paths.dataDirectory,'images');
conf.paths.logsDirectory    = fullfile(conf.paths.dataDirectory,'logs');

% Testing
conf.testSet          = 'test'; % {val, test}
conf.pe               = 0.01;    % probability of error for Chebysev bounds (*nLeaves for pe at root)
conf.useCascade       = 1;
conf.cascadeThresh    = -inf;   % threshold used to prune convolution locations (should not be changed)
conf.time             = 1;      % compare detection times
conf.onlyTypical      = 1;      % test only on typical aeroplane examples

% Empirical thresholds
conf.threshPerNode    = 1;      % use different threshold per each node (empirical)
                                % this should be set to 0 for computing the
                                % empirical thresholds
conf.empiricalThresh  = -0.5;   % global threshold for computing empirical thresholds
conf.empiricalPerc    = 0.95;   % use k% of highest scores per node to determine empirical threshold
conf.storeFullScores  = 0;      % store full scores (necessary for computing
                                % empirical thresholds