function view_jar_data(pos)

for i = 1:length(pos)
  im = imreadx(pos(i));
  showboxes(im, pos(i).boxes);
  hold on;
  plot([pos(i).polygon(1,:) pos(i).polygon(1,1)], ...
       [pos(i).polygon(2,:) pos(i).polygon(2,1)], ...
       '-', 'Color', 'r', 'LineWidth', 2);
  hold off;
  title(sprintf('%d/%d', i, length(pos)));
  %title(pos(i).im);
  %fprintf(sprintf('%s\n', pos(i).im));
  pause;
end
