#Task 1
print()
print("Task_1")
print()
#Data sets
bi_arr = []
se_arr = []
newt_arr = []
fx_arr = []


def f(x):
    return x**(3) - 3*x**(2) + x -1

def derivative_f(x):
    return 3*x**(2) - 6*x + 1

def g(x):
    return (3*x**2 -x +1)**(1/3)


def bisection(a, b, tol):
    if f(a) * f(b) >= 0:
       # print("Bisection method fails.")
        return None
    else:
        iterations = 0
        condition = True
        while  condition:  
            midpoint = (a + b) / 2
            bi_arr.append(f(midpoint))
            if f(midpoint) == 0:
                return midpoint
            elif f(a) * f(midpoint) < 0:
                b = midpoint
            else:
                a = midpoint
            
            condition = abs(f(midpoint)) >tol # the stop criteria
            #print(f"Iteration {iterations}: root = {midpoint:.10f} at function {f(midpoint)}")
            iterations += 1

            
        return midpoint

# Initial guesses for bisection method
a = 2
b =  3
tol = 1e-6

bisection(a, b, tol)
# print("Approximate root:", str(root)[:11])
# print( bi_farr)



def secant(x0, x1, tol):
    iterations = 0
    while abs(x1 - x0) > tol:
        x2 = x1 - (f(x1) * (x1 - x0)) / (f(x1) - f(x0))
        se_arr.append(f(x2))
        x0, x1 = x1, x2
        #print(f"Iteration {iterations}: root = {x2:.10f} funtion = {f(x2)}")
        iterations += 1
    return x2


# Initial guesses secant method;
x0 = -2
x1 = 3
tol = 1e-6

secant(x0, x1, tol)
# print("Approximate root:", str(root)[:11])
# print(se_arr)



def newton_raphson(x0, tol):
    iterations = 0
    while abs(f(x0)) >= tol:
        x1 = x0 - f(x0) / derivative_f(x0)
        newt_arr.append(f(x1))
        #print(f"Iteration {iterations}: root = {x1:.10f} funtion = {f(x1)}")
        iterations += 1
        x0 = x1
    return x1

# Initial guess Newton Raphs's method
x0 = 2
tol = 1e-6

newton_raphson(x0, tol)
# print("Approximate root:", str(root)[:11])
# print(n_arr)


def fixed_point_iteration(x0, tol):
    iterations = 0
    while True:
        x1 = g(x0)
        fx_arr.append(f(x1))
        #print(f"Iteration {iterations}: root = {x1:.10f} funtion = {f(x1)}")
        iterations += 1
        if abs(x1 - x0) < tol:
            break
        x0 = x1
    return x1

# Initial guess fixed_point_iteration
x0 = 2.0
tol = 1e-6

fixed_point_iteration(x0, tol)
# print("Approximate root:", str(root)[:11])
# print(fx_arr)
print()


#making the dat file
with open('./data/data.dat', 'w') as file:
    file.write("Iteration fx_Bisection fx_Secant fx_Newton fx_FixedPoint\n")
    
num_iterations = max(len(bi_arr), len(se_arr), len(newt_arr), len(fx_arr))
with open('./data/data.dat', 'a') as file:
    for i in range(num_iterations):

        output = f"{i}"
    
        # Bisection values
        if i < len(bi_arr):
            output += f" {bi_arr[i]}"
        else:
            output += " -"
        
        # Secant values
        if i < len(se_arr):
            output += f" {se_arr[i]}"
        else:
            output += " -"
        
        # Newton values
        if i < len(newt_arr):
            output += f" {newt_arr[i]}"
        else:
            output += " -"
        
        # Fixed-point values
        if i < len(fx_arr):
            output += f" {fx_arr[i]}"
        else:
            output += " -"
        
        file.write(f'{output}\n')
print("***************************************")
print('A data.dat file has been created')
print("***************************************")






#task 2
print()
print("Task_2")
print()
import numpy as np
import matplotlib.pyplot as plt

# Assuming a differential equation of the form dθ/dt = -k(θ - θ_env)
# where θ_env is the environmental temperature and k is a constant.
# This is a placeholder for the actual differential equation.

def dtheta_dt(theta, t) :
    return -2.2067 * 10**(-12) * (theta**4 - 81 * 10**(8))

def euler_method(dtheta_dt, theta0, t0, t_end, step):
    t_values = np.arange(t0, t_end + step, step)
    theta = np.zeros(len(t_values))
    theta = [theta0]
    
    for i in range(1, len(t_values)):
        theta.append(theta[i-1] + step * dtheta_dt(theta[i-1], t_values[i-1]))
    
    return t_values, theta

# Initial condition
theta0 = 1200  # Example initial temperature
t0 = 0  # Start time
t_end = 480  # End time

# Step sizes and exact solutions given
steps = [480, 240, 120, 60, 30]
exact_solutions = [1635.4, 537.26, 100.80, 32.607, 14.806]  # 

estimated_temp = []
print()

for i in range(len(steps)):
    step = steps[i]
    exact = exact_solutions[i] 
    t_values, theta = euler_method(dtheta_dt, theta0, t0, t_end, step)
    estimated_temp.append(theta[-1])
print("*********************************************************")
print("{:<15} {:<20} {:<20}".format("Step size", "Approximated solution", "Exact solution"))
for i in range(len(steps)):
    print("{:<15} {:<20.2f} {:<20.2f}".format(steps[i], estimated_temp[i], exact_solutions[i]))

# Writing results to a file
print("*********************************************************")

with open('./data/euler_method_results.txt', "w") as file:
    file.write("{:<15} {:<20} {:<20}\n".format("Step size", "Approximated solution", "Actual solution"))
    for i in range(len(steps)):
        file.write("{:<15} {:<20.2f} {:<20.2f}\n".format(steps[i], estimated_temp[i], exact_solutions[i]))
        
# Plotting
# Plot the results
plt.plot(steps, exact_solutions, marker='o', label="Actual Solution", color='green')
plt.plot(steps, estimated_temp, marker='o', label="Approximated Solution", color='red')
plt.xlabel("Step size (seconds)")
plt.ylabel(r"Temperature ($\theta$)")
plt.grid()
plt.title('Comparison of Euler\'s Method with Exact Solutions')
plt.grid(True)
plt.legend()

# Save plot to a file
plt.savefig('./plots/euler_vs_exact.png')








print()
#task 3
print("Task_3")	 
print()

#Coefficient Matrix.
A = np.array([[17,14,23], [-7.54,-3.54,2.7], [6,1,3]])
print("..........................")
print(A)
print("..........................")
print()
#Constant Vector.
y = np.array([24.5, 2.352, 14])
print("..........................")
print(y)
print("..........................")
print()
print()
#Using naive Gaussian elimination.

# Divide Row 1 by 17 and multipy it by -7.54.
mut_A = (A[0]/17)*-7.54

mut_y = (y[0]/17)*-7.54

# Subtract the result from Row 2 to get the resulting equation.
A[1] -= mut_A 

y[1] -= mut_y


# Divide Row 1 by 17 and multiply it by 6.
mut_A = (A[0]/17)*6

mut_y = (y[0]/17)*6

# Subtract the result from Row 3 to get the resulting equation.
A[2] -= mut_A

y[2] -= mut_y


# Use Row 2 as the pivot equation and eliminate Row 3

# Divide Row 2 by 2.669 and then multiply it by -3.941.
mut_A = (A[1]/2.669)*-3.941

mut_y = (y[1]/2.669)*-3.941

# Subtract the result from Row 3.

A[2] -= mut_A
A_rounded = np.round(A,3) 
print("..........................")
print(A_rounded)
print("..........................")
print()
y[2] -= mut_y 
y_rounded = np.round(y,3) 
print("..........................")
print(y_rounded)
print("..........................")
print()
print()
print()
	
print("--------------------------")
#Using back substitution to solve  the unknown values
n = len(y)
x = np.zeros(n)
	
for i in range(n-1, -1, -1):
     sum_term = 0
     for j in range(i+1, n):
          sum_term += A[i, j] * x[j]
     x[i] = (y[i] - sum_term) / A[i, i]
n = len(x)
for i in range(n):
    print(f"x_{i+1} = {x[i]:.3f}")
print("..........................")

	
