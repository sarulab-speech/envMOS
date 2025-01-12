#!/bin/bash
#PJM -L rscgrp=share-short
#PJM -L gpu=1
#PJM -L elapse=02:00:00
#PJM -g ge43
#PJM -j

module load cuda/12.1
uv run train.py