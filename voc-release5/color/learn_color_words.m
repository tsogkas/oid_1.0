function [C, A] = learn_color_words(pos, num_centers)

X = [];
for i = 1:length(pos)
  im = get_image(pos(i), 100);
  X = cat(2, X, im_to_col(im));
end

[C, A] = vl_kmeans(single(X), num_centers, 'verbose', ...
                   'algorithm', 'elkan', ...
                   'NumRepetitions', 5);

if num_centers == 30
  im = shiftdim(reshape(C, [3 3 10]), 1);
  figure;
  %subplot(1,2,1);
  %imagesc(imreadx(pos(1))); axis image;
  %subplot(1,2,2);
  imagesc(uint8(im)); axis image;
end


function im = get_image(pos, width)

im = imreadx(pos);
imsz = size(im);
y1 = min(max(1, round(pos.y1)), imsz(1));
y2 = min(max(1, round(pos.y2)), imsz(1));
x1 = min(max(1, round(pos.x1)), imsz(2));
x2 = min(max(1, round(pos.x2)), imsz(2));
im = im(y1:y2, x1:x2, :);
im = imresize(im, [nan width]);
