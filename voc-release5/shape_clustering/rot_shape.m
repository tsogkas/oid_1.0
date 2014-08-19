function pts = rot_shape(shape, rot, trans)
  % Rotate
  if nargin < 3
    mu = mean(shape, 1);
  else
    mu = trans;
  end
  pts = (shape - repmat(mu, size(shape,1),1));
  rad = rot/180*pi;
  R = [cos(rad) -sin(rad); ...
       sin(rad)  cos(rad)];
  pts = (R*pts')';
  pts = (pts + repmat(mu, size(shape,1),1));
end
