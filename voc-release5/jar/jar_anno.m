function anno = jar_anno()
% anno = jar_anno()
%   Returns the jar dataset annotations structure. 

conf = voc_config();

ld = load(conf.jar.anno);
anno = ld.anno;

% Create functions for testing object types given an object id
for i = 1:length(anno.objTypes)
  fn = @(id) (id >= (i-1)*anno.decoder) & (id < i*anno.decoder);
  t = anno.objTypes{i};
  eval(['anno.is_' anno.objTypes{i} ' = fn;']);
end

anno.get_type_id = @(id) 1+floor(id/anno.decoder);
anno.get_type = @(id) anno.objTypes{anno.get_type_id(id)};

% For consistency
if ~isfield(anno.aeroplane, 'imageId')
  anno.aeroplane.imageId = anno.aeroplane.parentId;
end
