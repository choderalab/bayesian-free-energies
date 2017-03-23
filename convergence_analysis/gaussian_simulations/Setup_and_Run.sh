#!/usr/bin/env bash

# Parameters for the Bayesian model and adaptive method
METHOD='thompson'
PRIOR=10

# Variables for the target problem
SMIN=1
SMAX=500
NSTATES=10

# Variables for the set of repeats
REPEATS=100
NITERATIONS=1000
MOVES=(1 2 5 10 20)

mkdir $METHOD
cd $METHOD
for M in ${MOVES[*]}; do
    ITERATIONS=$(expr $NITERATIONS / $M)
    mkdir ${M}_moves
    cd ${M}_moves
    sed "s/REPLACE/-r $REPEATS -m $METHOD --smin $SMIN --smax $SMAX --nstates $NSTATES --p_spread $PRIOR -c $ITERATIONS --save_freq $M --nmoves $M/" ../../submit_dummy > submit
    #qsub submit -N ${M}_moves
    cd ../
done
cd ../

