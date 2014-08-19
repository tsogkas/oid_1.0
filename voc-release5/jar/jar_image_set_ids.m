function ids = jar_image_set_ids(anno, image_set)

conf = voc_config();

image_set_path = [conf.jar.sets image_set '.txt'];
im_names = textread(image_set_path, '%s');

ids = anno.image.id(ismember(anno.image.name, im_names));
