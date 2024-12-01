#!/bin/bash

#Creating a folder to contain all the plots
mkdir -p data plots

#Enabling permissions
chmod 777 task.py && chmod 777 gnup.gp;

# Run the Python script
python3 4171220_project.py

# Run the Gnuplot script
gnuplot 4171220_plot.gp

echo '                                               '
echo '***********************************************'

echo 'A plot euler_vs_exact.png has been created'
echo 'A plot.png has also been created'

echo '***********************************************'
echo '                                               '







