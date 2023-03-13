#!/bin/bash

#cat=bowl
trial=4
for cat in bowl bottle
do 
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
done