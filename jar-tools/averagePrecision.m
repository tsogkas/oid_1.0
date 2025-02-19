function ap = averagePrecision(rec, prec)
% From the VOCdevkit (lazy)

rec = rec(:);
prec = prec(:);
mrec = [0; rec; 1];
mpre = [0; prec; 0];
for i = numel(mpre)-1:-1:1
  mpre(i) = max(mpre(i), mpre(i+1));
end
i = find(mrec(2:end) ~= mrec(1:end-1)) + 1;
ap = sum((mrec(i) - mrec(i-1)) .* mpre(i));
