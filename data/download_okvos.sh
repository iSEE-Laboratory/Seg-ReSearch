#!/bin/bash

#huggingface-cli download iSEE-Laboratory/OK_VOS \
#  --repo-type dataset \
#  --local-dir data/OK_VOS

tar -xzvf data/OK_VOS/train.tar.gz -C data/OK_VOS
#tar -xzvf data/OK_VOS/test.tar.gz -C data/OK_VOS

echo "Finish."