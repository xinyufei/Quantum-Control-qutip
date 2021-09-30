'''
This code was written primarily by Lucas Brady with inspiration from previous code
by Aniruddha Bapat and Stephen Jordan.
Additions were made by Ivy Liang
'''


import math
import numpy as np
#from matplotlib import pyplot as plt
from ctypes import *
from numpy import random as nrm
import random as rnd
from scipy.integrate import odeint
from scipy.interpolate import interp1d
from scipy.optimize import minimize
import networkx # for regular graphs
import os
import subprocess

Jij = False



############################################
# Functions to generate graph connectivity
############################################

def generate_Jij_LR(n, Zcoeff, alpha):
        '''Generates a Jij matrix for the long range Ising model'''
        global Jij
        
        Jij = np.zeros((n,n))
        for i in range(n):
                for j in range(i+1,n):
                    if (i!=j):
                        dist2 = (i-j)*(i-j);
                        Jij[i,j] = Zcoeff/(dist2**(0.5*alpha));
                        Jij[j,i] = Jij[i,j]

        return Jij


def generate_Jij_MC(n, d, seed=None):
        '''Generates a Jij matrix for n bits of MaxCut on a d-regular graph'''
        global Jij
        
        Jij = np.zeros((n,n))

        graph = networkx.random_regular_graph(d,n,seed=seed)
        
        edges=graph.edges()

            
        #edges = [(0, 2), (0, 3), (0, 5), (1, 4), (1, 6), (1, 7), (2, 5), (2, 7), (3, 5), (3, 6), (4, 6), (4, 7)]
            
        print(edges)
                
        for edge in edges:
            (i,j)=edge
            Jij[i,j] = 1
            Jij[j,i] = 1
         
        return Jij, edges
         
def generate_Jij(n, seed=None):
        '''Generates a randomized Jij matrix and stores it in a global variable'''
        global Jij
        
        Jij = np.zeros((n,n))
        for i in range(n):
                for j in range(i+1,n):
                        if seed:
                                nrm.seed((i*n*10+j*10)*seed*n*n*10+seed)
                        else:
                                nrm.seed(i*n*10+j*10)
                        Jij[i,j] = 2*nrm.rand() - 1
                        Jij[j,i] = Jij[i,j]
        return Jij
         




####################################
# Functions to generate Hamiltonian
####################################
def get_energy(x, Jij):
    n = len(Jij)
    val = 0
    for i in range(n):
        biti = int(x/(2**i))&1
        for j in range(i+1, n):
            bitj = int(x/(2**j))&1
            val = val + (Jij[i][j]*(1-(biti^bitj)*(2**1)))
    return val

def get_diag(Jij):
        '''Gets the diagonal of the cost function Hamiltonian.  This assumes
        you have already initialzed Jij'''
        
        #H = ham()
        n = len(Jij)
        #qc.allocateH(byref(H),n)
        diag = []
        for i in range(2**n):
            diag += [get_energy(i, Jij)]
        return diag


def flip_index (i, j, n):
        '''If i is the decimal version of a bit string of length n, this outputs
        the decimal version of the bit string that is that one but with
        bit j flipped'''
        
        rem = i
        one = +1
        for k in range(j+1):
                temp = rem - 2**(n-k-1)
                if (temp>=0):
                        rem =temp
                        one = -1
                else:
                        one = +1
        return i + one*2**(n-j-1)


def get_ham(n, want, Jij):
        '''Gets the Hamiltonian in a numpy format
        n=number of qubits
        want=boolean with True = C and False = B
        '''
        N = 2**n
        diag = get_diag(Jij)
        mat = []
        for i in range(N):
                unit = [0 for k in range(2*N)]
                unit[i] = 1
                if want:
                        col=applyC_sing(unit,diag)
                else:
                        col=applyB_sing(n, unit)
                mat += [[col[j]+1j*col[j+N] for j in range(N)]]
        return np.array(mat)
    
# works
def applyC_sing(y,diag):
        '''Applies the diagonal part of the Hamiltonian (i.e. C) to the vector y'''
        
        N = int(len(y)/2)
        output=[0 for i in range(2*N)]
        for i in range(N):
                output[i] = diag[i]*y[i]
                output[i+N] = diag[i]*y[i+N]
        return output
        
# works 
def applyB_sing(n, y):
        '''Applies the transverse field (i.e. B) to the vector y'''
        
        N = int(len(y)/2)
        output=[0 for i in range(2*N)]
        for i in range(N):
                for j in range(n):
                        index = flip_index(i,j,n)
                        output[i]   += -y[index] # real
                        output[i+N] += -y[index+N] # imaginary
        return output   













######################################################
# Generate Initial State
# The format is a 2*2**n list all with real elements
# followed by all imaginary elements
######################################################




def uniform(n):
        '''returns a list of length 2*2^n where the first 2^n entries are all
        sqrt(1/2^n) and the last ones are all 0
        This is usually the initial state'''
        N=2**n
        y = [1/math.sqrt(N) for i in range(N)]
        y += [0 for i in range(N)]
        
        return y







######################################
# Utility Functions
######################################


def get_u (t,uN,tf,ulist):
        '''interpolates the values of u stored in ulist to get the current value
        of u at t, there are uN values in ulist, in the order
        [u(0), u(tf/(uN-1)), ..., u(tf)]
        this function just does a linear interpolation'''
        
        if t>tf: t=tf
        
        lower = min(int(math.floor((uN)*(t/tf))), uN - 1);
        # upper = int(math.ceil((uN-1)*(t/tf)));
        # amount = (uN-1)*(t-tf*lower/(uN-1))/tf;
        #
        #
        # return (ulist[upper])*amount+(ulist[lower])*(1-amount);
        return ulist[lower]

def norm(y):
        '''returns the norm of the vector y where y has all the real 
        components in the first half and all the imaginary components
        in the second half'''
        output=0;
        N = len(y)/2
        for i in range(N):
                output+=y[i]**2+y[i+N]**2
        return output

def cdot(y1,y2):
        '''returns the complex inner product <y1|y2>, assumes the vectors
        have real components in the first half and imaginary components in the 
        second half'''
        output=0;
        N = int(len(y1)/2)
        for i in range(N):
                output+=(y1[i]-1j*y1[i+N])*(y2[i]+1j*y2[i+N])
        return output



#################################################
# ODE Solving of the Schrodinger equation
#################################################


def func_schro (y, t, n, uN, tf, ulist,diag) :
        '''This is the function f such that dy/dt = f(y,t), so this is essentially
        our differential equation or Schrodinger equation put into standard form.  
        t is the time variable
        y[] are the vector elements, all real parts first then all imaginaries
        f[] is the function f, and this will be the output
        *params is a pointer to an array of andy number of parameters we want
        This function assumes the form of B = -\sum \sigma_x
        and C is the Ising model with the defined Jij matrix'''
      
        N = 2**n
        
        u = get_u(t, uN, tf, ulist)
        
        dydt = [0 for i in range(2*N)]
        
        dydtC = applyC_sing(y,diag)

        dydtB = applyB_sing(n,y)

        for i in range(N):
            dydt[i] += u*dydtB[i+N] + (1-u)*dydtC[i+N]
            dydt[i+N] += -u*dydtB[i] - (1-u)*dydtC[i]
        
        """
        for i in range(N):
                # APPLY C
                dydt[i] = y[i+N]*diag[i]*(1-u); # real
                dydt[i+N] = -y[i]*diag[i]*(1-u);# imaginary
                # iterate over all "adjacent" states, i.e. one bit flip away 
                for j in range(n): # off-diagonal
                        # APPLY B
                        index = flip_index(i,j,n)
                        dydt[i] += -u*y[index+N] # real
                        dydt[i+N] += u*y[index] # imaginary
        """
        return dydt;

def func_schroN (y, t, n, uN, tf, ulist,diag) :
        '''This is the function f such that dy/dt = f(y,t), so this is essentially
        our differential equation put into standard form, running time in  
        t is the time variable
        y[] are the vector elements, all real parts first then all imaginaries
        f[] is the function f, and this will be the output
        *params is a pointer to an array of andy number of parameters we want
        This function assumes the form of B = -\sum \sigma_x
        and C is the Ising model with the defined Jij matrix
        This version is the negative and is used for reverse time evolution
        Note that we assume in this function that ulist has already been reversed 
        for the purposes of reverse evolution.'''
      
        N = 2**n
        
        u = get_u(t, uN, tf, ulist)
        
        dydt = [0 for i in range(2*N)]
        
        dydtC = applyC_sing(y,diag)
        
        dydtB = applyB_sing(n, y)
        for i in range(N):
            dydt[i] += -u*dydtB[i+N] - (1-u)*dydtC[i+N]
            dydt[i+N] += u*dydtB[i] + (1-u)*dydtC[i]
        """
        for i in range(N):
                dydt[i] = -y[i+N]*diag[i]*(1-u); # real
                dydt[i+N] = y[i]*diag[i]*(1-u);# imaginary
                # iterate over all "adjacent" states, i.e. one bit flip away 
                for j in range(n): # off-diagonal
                        index = flip_index(i,j,n)
                        dydt[i] += u*y[index+N] # real
                        dydt[i+N] += -u*y[index] # imaginary
        """
        return dydt;




#####################################################
# Functions to generate the analytic gradient
#####################################################

     

def avg_energy(y,diag):
        '''Tells us the energy expectation value of the state y
        At the moment, this just calculates the diagonal portion of the energy'''
        k = applyC_sing(y,diag)
        return cdot(y,k)

def get_k(yf, tlist, n, uN, tf, ulist, diag):
        '''Takes in the final value of the state yf and outputs the state k at all
        the time intervals given in tlist.  This uses our custom real first then
        imaginary in the second half vector form'''
        
        kf = applyC_sing(yf,diag)
        nulist = ulist[-1::-1]
        ntlist = tlist
        sol = odeint(func_schroN, kf, ntlist , args=(n,uN,tf,nulist,diag))
        return sol[-1::-1]


def get_Philist (tlist,n,tf,ulist,diag):
        '''Takes in a specific procedure, notably including the annealing
        path ulist and returns what the values of Phi are for that path
        at the times given by tlist
        Also returns the final energy of the procedure'''
        uN = len(ulist)
        y0 = uniform(n)

        all_y = odeint(func_schro, y0, tlist , args=(n,uN,tf,ulist,diag))
        
        #print "Figure of Merit: "+str(avg_energy(all_y[-1],diag))
        
        all_k = get_k(all_y[-1],tlist,n,uN,tf,ulist,diag)        
        
        Philist=[]
        for i in range(uN):
            Philist += [calc_Phi(all_y[i],all_k[i],n,diag)]
        
        #print(cdot(all_y[-1],all_y[-1]))
        return [Philist,np.real(avg_energy(all_y[-1],diag)),all_y]


def get_Philist_admm(tlist, n, tf, ulist, vlist, lambdalist, rho, diag):
        uN = len(ulist)
        y0 = uniform(n)

        all_y = odeint(func_schro, y0, tlist, args=(n, uN, tf, ulist, diag))

        # print "Figure of Merit: "+str(avg_energy(all_y[-1],diag))

        all_k = get_k(all_y[-1], tlist, n, uN, tf, ulist, diag)

        Philist = []
        norm_grad = np.zeros(uN)
        norm_grad[0] = rho * (ulist[1] - ulist[0] - vlist[0] + lambdalist[0])
        norm_grad[uN - 1] = rho * (ulist[uN - 1] - ulist[uN - 2] - vlist[uN - 2] + lambdalist[uN - 2])

        for t in range(1, uN - 1):
                norm_grad[t] = rho * (ulist[t] - ulist[t - 1] - vlist[t - 1] + lambdalist[t - 1])
        for i in range(uN):
                Philist += [calc_Phi(all_y[i], all_k[i], n, diag) + norm_grad[i]]

        # print(cdot(all_y[-1],all_y[-1]))
        return [Philist, np.real(avg_energy(all_y[-1], diag)), all_y]

def calc_Phi(y,k,n,diag):
        '''Calculates the value of Phi for the given y and k vectors
        This function assumes those vectors are for the same time and does not
        need any information about the time'''
        output = 0
        
        output +=  cdot(y,applyB_sing(n,k))
        output += -cdot(y,applyC_sing(k,diag))
        
        output = 2*np.imag(output)
        
        return output


def compute_energy_u(tlist, tf, ulist):
        global Jij
        n = len(Jij)
        diag = get_diag()
        return get_Energy_u(tlist, n, tf, ulist, diag)

def get_Energy_u (tlist,n,tf,ulist,diag):
        '''Takes in a specific procedure, notably including the annealing
        path ulist and returns what the value of the energy is for that path
        at the final time'''
        uN = len(ulist)
        y0 = uniform(n)
        
        all_y = odeint(func_schro, y0, tlist , args=(n,uN,tf,ulist,diag))
        
        return np.real(avg_energy(all_y[-1],diag))





#######################################################
# Carries out the gradient descent on the u(t) function
#######################################################

def compute_gradient(tlist, tf, ulist):
        global Jij
        n = len(Jij)
        diag = get_diag()
        [Philist, Energy, state] = get_Philist(tlist, n, tf, ulist, diag)
        return Philist
  

def gradient_descent_opt(n, uN, tf, iterations, min_grad, ulist_in=[], type="normal", v=None, _lambda=None, rho=None):
        '''Carries out the gradient descent and outputs the ulist from the end 
        of the procedure.
        n = number of qubits
        uN = number of points that u(t) is discretized into
        tf = the total time of the procedure
        iterations = how many times to do the gradient descent step
        ulist_in = intial guess for function, working on making a default, delete 
                and use internal code if you want something different
        Outputs:
                The final ulist
                Philist
                Final Energy'''        
        
        diag = get_diag() # Diagonal part of the Hamiltonian
        #diag = map(lambda x: diag[x],range(2**n))
        Etrue = min(diag)
        
        beta=250. # might need to up this number for more complicated procedures
        # could lower it for smaller systems to speed up convergence at the cost
        # of accuracy
        lambdas= 0
        
                
        if len(ulist_in)==0:
                # Use these as alternatives if you don' have an initial guess for ulist
                #ulist = map(lambda x: 1-x/(uN-1.), range(0,uN)) 
                ulist = list(map(lambda x: 0.5, range(0,uN)))   # this one works just fine
                #ulist = [nrm.rand() for i in range(uN)]    
        else:
                ulist=ulist_in
        tlist = list(map(lambda x: tf*x, map(lambda x: x/(uN-1.), range(0, uN))))
        
        ylist = ulist
        
        
        for i in range(iterations):
                lambdap = (1.+math.sqrt(1.+4.*lambdas**2))/2.
                
                gamma = (1-lambdas)/lambdap
                lambdas = lambdap

                if type == "admm":
                        [Philist, Energy, state] = get_Philist_admm(tlist, n, tf, ulist, v, _lambda, rho, diag)

                if type == "normal":
                        [Philist, Energy, state] = get_Philist(tlist, n, tf, ulist, diag)
                
                
                ylistnew = [max([0, min([1, ulist[j] + Philist[j]/(beta)])]) for j in range(uN)]
                
                ulist = [max([0, min([1, (1-gamma)*ylistnew[j]+gamma*ylist[j]])]) for j in range(uN)]

        
                ylist = ylistnew
        
                # print(str(tf)+"   "+str(i)+"/"+str(iterations)+": "+str([0+Energy,Etrue]))

                # print(np.linalg.norm(np.array(Philist), 2))

                # print(Philist)

                if np.linalg.norm(np.array(Philist), 2) < min_grad:
                        break
        num_it = i
        return [ulist, Philist, Energy, state, num_it]






##############################################
# IO Utility functions
##############################################



def import_u():
        '''Imports a previously found u(t).  I am mostly using this to improve previously found
        results and improve their quality'''
        infile=open("maxcut_ver2.tsv",'r') # change to your favorite file
        full = infile.read()
        infile.close()
        lines = full.split("\n")
        splitlines = map(lambda x: x.split("\t"), lines[:-1])
        numbers = [map(float,line) for line in splitlines]
        ulist = map(lambda x: x[2], numbers)
        qaoalist = map(lambda x: x[3], numbers)
        
        return [ulist,qaoalist]
     
 
def print_to_file(n,tf,tlist,ulist,Philist,Energy,edges):
        
        outstring = "B and C, n="+str(n)+", tf = "+str(tf)+"\n"
        outstring+= "Energy = "+str(Energy)+"\n"
        outstring+= str(edges)
        for i in range(len(ulist)):
                outstring+="\n"+str(tlist[i])+"\t"+str(ulist[i])+"\t"+str(Philist[i])
        outfile = open("B_and_C_tf="+str(tf)+"_n="+str(n)+".tsv",'w')
        outfile.write(outstring)
        print(ulist, outfile)
        outfile.close()
     
       
       
       
       
       
       
       
       
       
       
########################################
# What the program actually does
########################################
       
       

import sys
       
       
if __name__=="__main__":
        n = 6
        num_edges = 2
        seed = 2
        # Jij, edges = generate_Jij_MC(n, num_edges, 100)
        Jij = generate_Jij(n, seed)

        C_mat = get_ham(n, True, Jij)
        B_mat = get_ham(n, False, Jij)

        ######################################################
        # ... Sven's additions
        RealB = B_mat.real;
        ImagB = B_mat.imag;
        Brows, Bcols = np.nonzero(RealB)
        print("#nonzero REAL elements of B")
        for ii in range(len(Brows)):
                print("let RealB[",Brows[ii]+1,",",Bcols[ii]+1,"] := ",RealB[Brows[ii],Bcols[ii]],";")
        Brows, Bcols = np.nonzero(ImagB)
        print("#nonzero IMAGINARY elements of B")
        for ii in range(len(Brows)):
                print("let ImagB[",Brows[ii]+1,",",Bcols[ii]+1,"] := ",ImagB[Brows[ii],Bcols[ii]],";")
        RealC = C_mat.real;
        ImagC = C_mat.imag;
        Crows, Ccols = np.nonzero(RealC)
        print("#nonzero REAL elements of C")
        for ii in range(len(Crows)):
                print("let RealC[",Crows[ii]+1,",",Ccols[ii]+1,"] := ",RealC[Crows[ii],Ccols[ii]],";")
        Crows, Ccols = np.nonzero(ImagC)
        print("#nonzero IMAGINARY elements of C")
        for ii in range(len(Crows)):
                print("let ImagC[",Crows[ii]+1,",",Ccols[ii]+1,"] := ",ImagC[Crows[ii],Ccols[ii]],";")
        ######################################################
