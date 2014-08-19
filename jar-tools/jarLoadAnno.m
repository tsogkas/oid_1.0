function anno = jarLoadAnno()
% anno = jarLoadAnno()
%
% Load the jar annotation structure augmented with some useful fields.

% conf = jarConf();
% load(conf.path.anno);
load anno.mat

if ~isfield(anno.aeroplane, 'imageId')
  anno.aeroplane.imageId = anno.aeroplane.parentId;
end

% Compute bounding boxes from polygons for each object type
for i = 1:length(anno.meta.objTypes)
  objType = anno.meta.objTypes{i};
  if isfield(anno, objType) && isfield(anno.(objType), 'polygon')
    box = cellfun(@polygonToBox, anno.(objType).polygon, 'UniformOutput', false);
    anno.(objType).box = cat(2, box{:});
  end
end

% Compute object counts per image
nImages = length(anno.image.id);
anno.image.objCount.aeroplane = zeros(1, nImages);

% imageInds are ints in [1, length(anno.image.id)]
imageInds = vl_binsearch(anno.image.id, anno.aeroplane.imageId);
counts = histc(imageInds, (0:length(anno.image.id)) + 0.5);
anno.image.objCount.aeroplane = counts(1:end-1);

%[[[ test that counts are correct
%assert(length(anno.image.id) == length(anno.image.objCount.aeroplane));
%for i = 1:length(anno.image.id)
%  nAeros = anno.image.objCount.aeroplane(i);
%  imageId = anno.image.id(i);
%  assert(nAeros == length(find(anno.aeroplane.imageId == imageId)));
%end
%]]]

%[[[ determine attribute labels and 'is difficult' flags
% Any attribute with less than 80% confidence from turkers is flagged
% as difficult
attrVoteThresh = 0.8;

for i = 1:length(anno.meta.objTypes)
  objType = anno.meta.objTypes{i};
  if ~isfield(anno, objType), continue, end;

  anno.(objType).isDifficult = false(1, length(anno.(objType).id));

  if ~isfield(anno.(objType), 'attribute'), continue, end;

  fns = fieldnames(anno.(objType).attribute);
  for j = 1:length(fns)
    fn = fns{j};
    [v, l] = max(anno.(objType).attribute.(fn));
    anno.(objType).attribute.([fn 'IsDifficult']) = (v < attrVoteThresh);
    anno.(objType).attribute.([fn 'Label']) = l;
  end
end
%]]]


anno.getTypeId = @(id) 1+floor(id/anno.meta.decoder);
anno.getType = @(id) anno.meta.objTypes{anno.getTypeId(id)};

%[[[ add imageId arrays
anno.verticalStabilizer.imageId = zeros(1,length(anno.verticalStabilizer.id));
anno.nose.imageId = zeros(1,length(anno.nose.id));
anno.wing.imageId = zeros(1,length(anno.wing.id));
anno.wheel.imageId = zeros(1,length(anno.wheel.id));

for i = 1:length(anno.image.id)
  imageId = i;
  x = find(anno.aeroplane.parentId == imageId);
  for j = 1:length(x)
    objectId = anno.aeroplane.id(x(j));
    y = find(anno.verticalStabilizer.parentId == objectId);
    anno.verticalStabilizer.imageId(y) = anno.image.id(i);
    y = find(anno.nose.parentId == objectId);
    anno.nose.imageId(y) = anno.image.id(i);
    y = find(anno.wing.parentId == objectId);
    anno.wing.imageId(y) = anno.image.id(i);
    y = find(anno.wheel.parentId == objectId);
    anno.wheel.imageId(y) = anno.image.id(i);
  end
end
%]]]

%[[[ synthesize wheelPhrase and wingPhrase examples
newTypes = {'wheelPhrase', 'wheel', 'wingPhrase', 'wing'};
for t = 1:2:length(newTypes)
  newType = newTypes{t};
  memberType = newTypes{t+1};
  anno.meta.objTypes{end+1} = newType;
  anno.meta.(newType).memberType = memberType;
  count = 0;
  typeIdx = length(anno.meta.objTypes)-1;
  for i = 1:length(anno.aeroplane.id)
    aeroId = anno.aeroplane.id(i);
    imageId = anno.aeroplane.imageId(i);
    y = find(anno.(memberType).parentId == aeroId);
    if isempty(y), continue, end;

    boxes = anno.(memberType).box(:,y);
    box = [min(boxes(1:2,:),[],2); max(boxes(3:4,:),[],2)];
    poly = [box([1 1 3 3 1])'; ...
            box([2 4 4 2 2])'];

    count = count + 1;
    anno.(newType).id(count) = anno.meta.decoder*typeIdx + count;
    anno.(newType).parentId(count) = aeroId;
    anno.(newType).polygon{count} = poly;
    anno.(newType).box(:,count) = box;
    anno.(newType).isDifficult(count) = any(anno.(memberType).isDifficult(y));
    anno.(newType).imageId(count) = imageId;
    anno.(newType).members{count} = y;
  end
end
%]]]



% ------------------------------------------------------------------------
function box = polygonToBox(polygon)
% ------------------------------------------------------------------------
box = [min(polygon(1,:)); min(polygon(2,:)); ...
       max(polygon(1,:)); max(polygon(2,:))];
