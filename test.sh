#!/bin/bash
export srcf=$1
export rsf=$2
export modf=$3
cd mkdata
bash mktest.sh $1
cd ..
th pred.lua "$3" $2
