#!/bin/bash
# Run by qsub_jar_train.pl

cd /home/rbg/rel5-dev

/export/ws12/tduosn/software/matlab2011a/bin/matlab -nodisplay -nodesktop << EOF
matlabpool open $3;
jar_train_from_em_hog_clusters('$1', true, $2);
EOF

exit 0
