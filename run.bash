#!/bin/bash

cat=cylinder
trial=2
for i in {0..19}
do
    python process_main.py --cat $cat --trial $trial --idx $i --sim_device cuda:0
    ret=$?
    while [ $ret -eq 0 ];
    do
        python process_chunk.py --cat $cat --trial $trial --idx $i --sim_device cuda:0
        ret=$?
    done
done