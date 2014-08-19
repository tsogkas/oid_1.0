function pos_montage(pos, width)

h = -inf;
for i = 1:length(pos)
  im{i} = get_image(pos(i), width);
  h = max(h, size(im{i}, 1));
end

for i = 1:length(pos)
  hi = size(im{i}, 1);
  im{i} = padarray(im{i}, [h-hi 0 0], 0, 'post');
end

montage(cat(4, im{:}), 'size', [nan round(sqrt(length(im)))]);



function im = get_image(pos, width)

im = imreadx(pos);
imsz = size(im);
y1 = min(max(1, round(pos.y1)), imsz(1));
y2 = min(max(1, round(pos.y2)), imsz(1));
x1 = min(max(1, round(pos.x1)), imsz(2));
x2 = min(max(1, round(pos.x2)), imsz(2));
im = im(y1:y2, x1:x2, :);
im = imresize(im, [nan width]);
