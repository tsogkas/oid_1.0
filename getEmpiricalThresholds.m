% getEmpiricalThresholds
% Use a percentage of highest scoring positive examples to construct
% empirical thresholds for each node of a filter tree hierarchy.
% 
%       [t,info] = getEmpiricalThresholds(totalScores, globalThresh, levelScores, percentage)
% 
% INPUT:
%   totalScores:  scores for all positives in the trainval set
%   globalThresh: global threshold used to filter out high-scoring positives
%   levelScores:  scores for every node of the hierarchy
%   percentage:   we use the percentage% highest scoring positives 
%                 corresponding to each node to determine the empirical thresholds 
% 
% Stavros Tsogkas <stavros.tsogkas@ecp.fr>
% Last update: June 2014

function [t,info] = getEmpiricalThresholds(totalScores, globalThresh, levelScores, percentage)

% use only positives with score >= globalThresh
highScores          = totalScores >= globalThresh;
levelScoresCropped  = levelScores;
levelScoresCropped(:,sum(levelScoresCropped,1)==0) = [];
levelScoresCropped  = levelScoresCropped(:,highScores);

% use a percentage of scores at each node, to avoid outliers. 
isInf = isinf(levelScoresCropped);  
nBins = 100;
scoresNotInf = cell(size(isInf,1),1);
nElements    = cell(size(isInf,1),1);
xCenters     = cell(size(isInf,1),1);
cumPerc      = cell(size(isInf,1),1);
ind          = cell(size(isInf,1),1);
t            = zeros(size(isInf,1),1);
for i=1:size(levelScoresCropped,1)
    scoresNotInf{i} = levelScoresCropped(i,~isInf(i,:));
    [nElements{i},xCenters{i}] = hist(scoresNotInf{i},nBins);
    cumPerc{i} = cumsum(nElements{i})/sum(nElements{i});
    ind{i} = find(cumPerc{i} <= 1-percentage);
    if isempty(ind{i})
        t(i) = min(scoresNotInf{i});
    else
        t(i) = xCenters{i}(ind{i}(end));
    end
end
info.scoresNotInf = scoresNotInf;
info.nElements = nElements;
info.xCenters = xCenters;
info.cumPerc = cumPerc;
info.ind = ind;