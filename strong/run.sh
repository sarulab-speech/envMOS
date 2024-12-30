#!/bin/bash
#PJM -L rscgrp=share-debug
#PJM -L gpu=1
#PJM -L elapse=00:30:00
#PJM -g ge43
#PJM -j

module load cuda/12.1
uv run train.py