#!/bin/bash

bash mkhglove.sh data/sp-chi-data1/pd-pwb.obj.train.30.5.final.txt ch-data1
bash mkglove.sh data/sp-chi-data2/sp-chi-data2.txt ch-data2
bash mkglove.sh data/sp-eng-data1/afp.train.final.txt en-afp
bash mkglove.sh data/sp-eng-data1/nyt.train.final.txt en-nyt

