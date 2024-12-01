# Set the terminal type and output file
set terminal png size 2035,1024
set output "./plots/plot.png"

set title "iterations vrs iterative f(x)s"
set xlabel "iterations"
set ylabel "iterative method's f(x)"
# Plot the data from data.dat using specified columns and plot settings
plot "./data/data.dat" using 1:2 title "Iteration vs Bisection" w lp,\
     "" using 1:3 title "Iteration vs Secant" lt 8 lc 12 w lp,\
     "" using 1:4 title "Iteration vs Newton" lt 4 lc 2 w lp,\
     "" using 1:5 title "Iteration vs Fixed Iter" lt 3 lc 6 w lp



	set terminal png size 800, 600
	set output "./plots/euler_gp.png"
	set xlabel "Step Size: Time(sec)"
	set ylabel "0"
	set title "Approximate and Actual Solution for an ODE: Euler"
	set key left
	set grid

	plot "./data/euler_method_results.txt" using 1:2 title "Euler Approximate" lt 22 dt 1 lw 3 lc 25 w lp, "" using 1:3 title "Actual Value" lt 24 dt 1 lw 3 lc 7 w lp



