% JAR_TEST  Evaluates the hierarchical cascade introduced in [1] on the OID
% test set.
%           
% [1] Understanding Objects in Detail with Fine-grained Attributes (Vedaldi
% et al. CVPR 2014)
% 
%       [ap, recall, prec] = jar_test(model, anno, obj_class, image_sets,...
%                                       only_typical, note)
% 
% INPUT:
%   model:        The model to be evaluated.
%   anno:         Annotation struct.
%   obj_class:    The object class ('aeroplane' in our case).
%   image_sets:   'val' or 'test'.
%   only_typical: Use only typical aeroplane examples.
%   note:         A short description of the model.
% 
% Stavros Tsogkas <stavros.tsogkas@ecp.fr> (modified R. B. Girshick's code)
% Last update: June 2014


function [ap, recall, prec] = jar_test(model, anno, obj_class, ...
    image_sets, only_typical, note)

if ~exist('infix', 'var'), note = ''; end
if ~iscell(image_sets), image_sets = {image_sets}; end


scale = 0.5;
oidconf = oidConfig();
jar_conf = jarConf();
[image_ids, tag] = jarReadImageSet(anno, obj_class, image_sets, only_typical);
image_inds = find(ismember(anno.image.id, image_ids));
useCascade = oidconf.useCascade;
savePath = fullfile(oidconf.paths.resultsDirectory,...
    [model.class '_' note '_detections_' tag,...
    '_pe' num2str(oidconf.pe) '_t' num2str(oidconf.cascadeThresh) '.mat']);

try
    load(savePath)
catch
    parfor i = 1:length(image_inds)
        fprintf('%s: testing: %s, %d/%d\n', model.class, image_sets{:}, ...
            i, length(image_inds));
        
        image_ind = image_inds(i);
        image_name = anno.image.name{image_ind};
        im = imread([jar_conf.path.image image_name]);
        im = imresize(im, scale);
        
        [ds, bs] = imgdetect(im, model, model.thresh,useCascade);
        
        if ~isempty(bs)
            %unclipped_ds = ds(:, 1:4);
            [ds, bs, rm] = clipboxes(im, ds, bs);
            %unclipped_ds(rm, :) = [];
            
            % NMS
            I = nms(ds, 0.5);
            ds = ds(I,:);
            bs = bs(I,:);
            %unclipped_ds = unclipped_ds(I,:);
            ndets = size(ds, 1);
            
            % Rescale boxes
            ds(:, 1:4) = (ds(:, 1:4) - 1) / scale + 1;
            %unclipped_ds(:, 1:4) = (unclipped_ds(:, 1:4) - 1) / scale + 1;
            bs = reduceboxes(model, bs);
            bs(:, 1:end-2) = (bs(:, 1:end-2) - 1) / scale + 1;
            
            dets(i).box          = ds(:, 1:4)';
            dets(i).score        = ds(:, end)';
            dets(i).image_id     = repmat(image_ids(i), [1 ndets]);
            dets(i).component    = ds(:, end-1)';
            dets(i).filter_boxes = bs';
        else
            dets(i).box          = [];
            dets(i).score        = [];
            dets(i).image_id     = [];
            dets(i).component    = [];
            dets(i).filter_boxes = [];
        end
    end
    
    det.objType     = model.class;
    det.imageSets   = image_sets;
    det.objClass    = obj_class;
    det.onlyTypical = only_typical;
    det.box         = cat(2, dets(:).box);
    det.score       = cat(2, dets(:).score);
    det.imageId     = cat(2, dets(:).image_id);
    det.component   = cat(2, dets(:).component);
    det.filterBoxes = cat(2, dets(:).filter_boxes);
    
    [ap, recall, prec] = jarEvalDetections(det, anno);
    
    save(savePath, 'det', 'ap', 'recall', 'prec', 'oidconf');
end