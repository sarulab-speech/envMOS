#!/bin/bash
#PJM -L rscgrp=regular-a
#PJM -L node=1
#PJM -L elapse=48:00:00
#PJM -g ge43
#PJM -j

module load cuda/12.1
uv run train.py