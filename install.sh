#!/bin/bash

export PYTHONNOUSERSITE=1
wget -qO- https://micromamba.snakepit.net/api/micromamba/linux-64/latest | tar -xvj bin/micromamba
bin/micromamba create -f environment/env.yml -r runtime -n tts -y
bin/micromamba create -f environment/env.yml -r runtime -n tts -y
exit
