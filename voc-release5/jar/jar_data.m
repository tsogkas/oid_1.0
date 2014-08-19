function [pos, neg, impos] = jar_data(obj_type, obj_class, image_sets, only_typical)

conf       = voc_config();
jar_conf   = jarConf();
oidconf    = oidConfig();
% cachedir   = conf.paths.model_dir;
cachedir   = [oidconf.paths.dataDirectory '/'];
scale      = conf.jar.train_scale;

anno = jarLoadAnno();
% Get image ids listed in the image set
[im_ids, tag] = jarReadImageSet(anno, obj_class, image_sets, only_typical);

cache_file = [cachedir obj_type '_jar_data_from_' tag];
try
  load(cache_file);
catch
  % Remove images with more than one annotated aeroplane
  I = find(anno.image.objCount.aeroplane > 1);
  rm_im_ids = anno.image.id(I);
  im_ids = setdiff(im_ids, rm_im_ids);

  % Get objects belonging to the selected image
  sel_objs = find(ismember(anno.(obj_type).imageId, im_ids));
  num = length(sel_objs);

  pos      = [];
  impos    = [];
  numpos   = 0;
  numimpos = 0;
  dataid   = 0;

  for i = 1:num
    tic_toc_print('%s: parsing positives (%s): %d/%d\n', ...
                  obj_type, [image_sets{:}], i, num);

    ind = sel_objs(i);
    image_id = anno.(obj_type).imageId(ind);
    image_ind = vl_binsearch(anno.image.id, image_id);

    image_path = [jar_conf.path.image anno.image.name{image_ind}];

    aero_ind = get_aero(anno, obj_type, ind);
    if isempty(aero_ind)
      continue;
    end

    numpos = numpos + 1;
    dataid = dataid + 1;
    poly   = anno.(obj_type).polygon{ind};
    poly   = (poly-1)*scale + 1;
    bbox   = [min(poly(1,:)) min(poly(2,:)) max(poly(1,:)) max(poly(2,:))];
    sz     = round(anno.image.size(:,image_ind) * scale);
    
    pos(numpos).scale   = scale;
    pos(numpos).im      = image_path;
    pos(numpos).x1      = bbox(1);
    pos(numpos).y1      = bbox(2);
    pos(numpos).x2      = bbox(3);
    pos(numpos).y2      = bbox(4);
    pos(numpos).boxes   = bbox;
    pos(numpos).flip    = false;
    pos(numpos).dataids = dataid;
    pos(numpos).sizes   = (bbox(3)-bbox(1)+1)*(bbox(4)-bbox(2)+1);
    pos(numpos).polygon = poly;
    %pos(numpos).anno    = soaSubsRef(anno.(obj_type), ind);
    pos(numpos).facing  = anno.aeroplane.attribute.facingDirection(:,aero_ind);
    pos(numpos).anno_ind= ind;
    pos(numpos).obj_type= obj_type;

    % Create flipped example
    numpos    = numpos + 1;
    dataid    = dataid + 1;
    oldx1     = bbox(1);
    oldx2     = bbox(3);
    bbox(1)   = sz(2) - oldx2 + 1;
    bbox(3)   = sz(2) - oldx1 + 1;
    poly(1,:) = sz(2) - poly(1,:) + 1;
    facing    = pos(numpos-1).facing([5 4 3 2 1 8 7 6]);

    pos(numpos).scale   = scale;
    pos(numpos).im      = image_path;
    pos(numpos).x1      = bbox(1);
    pos(numpos).y1      = bbox(2);
    pos(numpos).x2      = bbox(3);
    pos(numpos).y2      = bbox(4);
    pos(numpos).boxes   = bbox;
    pos(numpos).flip    = true;
    pos(numpos).dataids = dataid;
    pos(numpos).sizes   = (bbox(3)-bbox(1)+1)*(bbox(4)-bbox(2)+1);
    pos(numpos).polygon = poly;
    %pos(numpos).anno    = soaSubsRef(anno.(obj_type), ind);
    pos(numpos).facing  = facing;
    pos(numpos).anno_ind= ind;
    pos(numpos).obj_type= obj_type;
  end
  impos = pos;
  save(cache_file, 'pos', 'impos');
end

% Get negatives from pascal 2007 aeroplane negatives
[~, neg] = pascal_data('aeroplane', '2007');



function ind = get_aero(anno, obj_type, ind)

while ~strcmp(obj_type, 'aeroplane')
  if ~isfield(anno.(obj_type), 'parentId')
    ind = [];
    break;
  end
  parent_id = anno.(obj_type).parentId(ind);
  obj_type = anno.getType(parent_id);
  ind = vl_binsearch(anno.(obj_type).id, parent_id);
end

assert(isempty(ind) || strcmp(obj_type, 'aeroplane'));
