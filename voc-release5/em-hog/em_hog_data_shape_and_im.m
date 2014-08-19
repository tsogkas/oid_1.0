function [X, num_xforms] = em_hog_data_shape_and_im(pos)

conf = voc_config();

obj_type = pos(1).obj_type;

param.TPT_SIZE = conf.em_hog.(obj_type).tpt_size;
param.CELL_SIZE = 8;
param.THRESH = 0.2;

num_pos = length(pos);
dim = conf.features.dim*param.TPT_SIZE^2;

xspace = conf.em_hog.(obj_type).x_space;
yspace = conf.em_hog.(obj_type).y_space;
sspace = conf.em_hog.(obj_type).s_space;
num_xforms = length(xspace)*length(yspace)*length(sspace);

X = sparse(dim, num_xforms*num_pos);
pos_ind = 1;
for i = 1:num_xforms:num_xforms*num_pos
  j = 0;
  for ds = sspace
    for dx = xspace
      for dy = yspace
        param.DX = dx;
        param.DY = dy;
        param.BASE_SCALE = ds;
        tic_toc_print('%d/%d\n', pos_ind, num_pos);
        f = extract_feature(pos(pos_ind), param);
        X(:,i+j) = f(:);
        j = j + 1;
      end
    end
  end
  pos_ind = pos_ind + 1;
end


%function f = extract_feature(pos, param)
%
%box = pos.boxes;
%width = box(3)-box(1)+1;
%height = box(4)-box(2)+1;
%scale = param.BASE_SCALE*param.CELL_SIZE*param.TPT_SIZE/max(width,height);
%box = (box-1)*scale+1;
%box([1 3]) = box([1 3]) + param.DX*param.CELL_SIZE;
%box([2 4]) = box([2 4]) + param.DY*param.CELL_SIZE;
%
%HALF_SIZE = param.TPT_SIZE*param.CELL_SIZE/2;
%width = box(3)-box(1)+1;
%height = box(4)-box(2)+1;
%cx = box(1) + width/2;
%cy = box(2) + height/2;
%outer_box = [cx-HALF_SIZE cy-HALF_SIZE cx+HALF_SIZE cy+HALF_SIZE];
%outer_box = round(outer_box);
%box = round(box);
%
%pos.scale = pos.scale*scale;
%im = imreadx(pos);
%im = double(im);
%
%im = subarray(im, outer_box(2), outer_box(4), ...
%              outer_box(1), outer_box(3), 1);
%
%box(3:4) = box(3:4) - box(1:2) + 1 + box(1:2) - outer_box(1:2);
%box(1:2) = 1 + box(1:2) - outer_box(1:2);
%outer_box(3:4) = outer_box(3:4) - outer_box(1:2) + 1;
%outer_box(1:2) = 1;
%
%f = features2(im, param.CELL_SIZE);
%f = f > param.THRESH;
%
%subplot(2,1,1);
%fprintf('%d %d\n', round(box(3)-box(1)+1), round(box(4)-box(2)+1));
%showboxes(uint8(im), cat(1, box, outer_box));
%subplot(2,1,2);
%visualizeHOG(max(0, f));
%pause;

function f = extract_feature(pos, param)

box = pos.boxes;
shape = pos.polygon;
width = box(3)-box(1)+1;
height = box(4)-box(2)+1;
scale = param.BASE_SCALE*param.CELL_SIZE*param.TPT_SIZE/max(width,height);
box = (box-1)*scale+1;
box([1 3]) = box([1 3]) + param.DX*param.CELL_SIZE;
box([2 4]) = box([2 4]) + param.DY*param.CELL_SIZE;

mu = mean(shape, 2);
shape = bsxfun(@minus, shape, mu);
shape = (shape-1)*scale+1;
shape = bsxfun(@plus, shape, mu*scale);

HALF_SIZE = param.TPT_SIZE*param.CELL_SIZE/2;
width = box(3)-box(1)+1;
height = box(4)-box(2)+1;
cx = box(1) + width/2;
cy = box(2) + height/2;
outer_box = [cx-HALF_SIZE cy-HALF_SIZE cx+HALF_SIZE cy+HALF_SIZE];
outer_box = round(outer_box);
box = round(box);

%pos.scale = pos.scale*scale;
%im = imreadx(pos);
%im = double(im);

pos_scale = pos.scale*scale;
sob = outer_box;
mu = [sob(1) + (sob(3)-sob(1))/2 ...
      sob(2) + (sob(4)-sob(2))/2];
sob = sob - [mu mu];
sob = (sob-1)/pos_scale+1;
sob = sob + [mu mu]/pos_scale;
sob = round(sob);
pos = rmfield(pos, 'scale');
im = imreadx(pos);
im = double(im);
im = subarray(im, sob(2), sob(4), ...
              sob(1), sob(3), 1);

%im = subarray(im, outer_box(2), outer_box(4), ...
%              outer_box(1), outer_box(3), 1);

box(3:4) = box(3:4) - outer_box(1:2) + 1;
box(1:2) = 1 + box(1:2) - outer_box(1:2);
shape(1,:) = shape(1,:) - outer_box(1) + 1;
shape(2,:) = shape(2,:) - outer_box(2) + 1;
outer_box(3:4) = outer_box(3:4) - outer_box(1:2);
outer_box(1:2) = 1;

im = imresize(im, outer_box([4 3]), 'bilinear');
im_sz = size(im);

boundary_im = zeros(im_sz(1:2));

mask = roipoly(boundary_im, shape(1,:), shape(2,:));
im = im .* repmat(mask, [1 1 3]);

shape = shape';
boundary = poly2boundary(shape);
out_of_image = boundary(:,1) < 1 | boundary(:,1) > im_sz(2) | boundary(:,2) < 1 | boundary(:,2) > im_sz(1);
boundary = boundary(~out_of_image,:);
boundary_idx = sub2ind(im_sz, boundary(:,2), boundary(:,1));
boundary_im(boundary_idx) = 1;

im = im + 255*repmat(boundary_im, [1 1 3]);

boundary_im = repmat(boundary_im, [1 1 3]);
f = features2(im, param.CELL_SIZE);
f = f > param.THRESH;

%subplot(2,1,1);
%fprintf('%d %d\n', round(box(3)-box(1)+1), round(box(4)-box(2)+1));
%showboxes(uint8(im), cat(1, box, outer_box));
%subplot(2,1,2);
%visualizeHOG(max(0, f));
%pause;
