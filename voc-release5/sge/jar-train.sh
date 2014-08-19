#!/bin/bash
# Run by qsub_jar_train.pl

cd /home/rbg/rel5-dev

/export/ws12/tduosn/software/matlab2011a/bin/matlab -nodisplay -nodesktop << EOF
%disp(pwd);
fprintf('\n\nTraining: $1\n\n');
matlabpool open 4;
%jar_train('$1', 3);
jar_train_facing_mixture('aeroplane');
EOF

exit 0
