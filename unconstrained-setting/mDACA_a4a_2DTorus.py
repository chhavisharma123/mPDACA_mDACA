#!/usr/bin/env python
# coding: utf-8

# # Import required packages

# In[ ]:


from matplotlib.collections import EventCollection
import numpy as np
import copy
import random as rd
import networkx as nx
import math
import matplotlib.pyplot as plt
from networkx.algorithms.bipartite.generators import complete_bipartite_graph
from numpy import linalg as LA
import csv
from sklearn.utils import shuffle


# In[ ]:


"""
Download data set a4a from LIBSVM
Website: https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html
Load csv file of downloaded data
"""

mydata = np.genfromtxt('a4a.csv',delimiter = ',')
num_features = len(mydata[0]) - 1
num_samples = len(mydata)
print('num samples: %d num features : %d ' %(num_samples, num_features))


# In[ ]:


"""
Shuffling the indices
"""

indices = np.arange(len(mydata))
# print (indices)
np.random.seed(100)
np.random.shuffle(indices)
print ('indices after shuffling',indices)
# print (mydata[0])
shuffled_data = [list(mydata[i]) for i in indices]


# In[ ]:


"""
Separating features and labels from shuffled data
"""
shuffled_data = np.array(shuffled_data)
X1 = shuffled_data[:,1:(num_features + 1)]
Y1 = shuffled_data[:,:1]
print (len(X1[0]))
X = [X1[i]/LA.norm(X1[i]) for i in range(len(X1))]
Y = [Y1[i][0] for i in range(len(Y1)) ]


# In[ ]:


"""
Adding 1 to each feature vector to incorporate offset parameter
"""
X = [ np.insert(X[i],0,1) for i in range(len(X))]
d1 = len(X)
print (len(X[0]))

print (len(X))
print (len(Y))


# In[ ]:


def data_blocks(X,Y,m):
    """
    This function distributes features and corresponding lables to m nodes

    Input
    ---------
     X: feature matrix, Y: labels, m: number of nodes"

    Returns
     --------
    features: A list containing features of all nodes
    values: A list containing associated labels of all nodes

    """
    BX = [[] for i in range(m)]
    BY = [[] for i in range(m)]
    d1 = len(X)
    s2 = int(d1/m)  ## size of each block
    g = 0
    for i in range(m):
        for j in range(g,s2 + g,1):
            BX[i].append(X[j])
            BY[i].append(Y[j])
        g = g + s2 
    samples_after_eq_dist = m*int(d1/m)  ## total samples after equally distributed samples to every node
    remaining_samples = d1 - m*int(d1/m) ### This quantity will always be less than equal to m
    if ( remaining_samples >= 1):
        for j in range(remaining_samples):
            BX[j].append(X[samples_after_eq_dist + j])
            BY[j].append(Y[samples_after_eq_dist + j])

    return [BX,BY]  


# # Generate 2D Torus topology

# In[ ]:


def empty_graph(n=0,create_using=None):
    """Return the empty graph with n nodes and zero edges.

    Node labels are the integers 0 to n-1
    
    """
    
    if create_using is None:
        # default empty graph is a simple graph
        G=nx.Graph()
    else:
        G=create_using
        G.clear()

    G.add_nodes_from(range(n))
    G.name="empty_graph(%d)"%n
    return G


def grid_2d(m,n,periodic=False,create_using=None): ## m,n be the number of rows and number of 
    # columns in torus topolgy
    
    """ Return the 2d grid graph of mxn nodes,
        each connected to its nearest neighbors.
        Optional argument periodic=True will connect
        boundary nodes via periodic boundary conditions.
    """
    
    G=empty_graph(0,create_using)
    G.name="grid_2d_graph"
    rows=range(m)
    columns=range(n)
    G.add_nodes_from( (i,j) for i in rows for j in columns )
    G.add_edges_from( ((i,j),(i-1,j)) for i in rows for j in columns if i>0 )
    G.add_edges_from( ((i,j),(i,j-1)) for i in rows for j in columns if j>0 )
    if G.is_directed():
        G.add_edges_from( ((i,j),(i+1,j)) for i in rows for j in columns if i<m-1 )
        G.add_edges_from( ((i,j),(i,j+1)) for i in rows for j in columns if j<n-1 )
    if periodic:
        if n>2:
            G.add_edges_from( ((i,0),(i,n-1)) for i in rows )
            if G.is_directed():
                G.add_edges_from( ((i,n-1),(i,0)) for i in rows )
        if m>2:
            G.add_edges_from( ((0,j),(m-1,j)) for j in columns )
            if G.is_directed():
                G.add_edges_from( ((m-1,j),(0,j)) for j in columns )
        G.name="periodic_grid_2d_graph(%d,%d)"%(m,n)
    return G


# # Computing weight matrix W associated with 2D-Torus topology

# In[ ]:


def gen_graph(row,column,m,w1,w2):
    
    """
    row = column + 1
    m: number of nodes
    w1,w2: weights to generate entries of W 
    
    """

    A = [[0 for j in range(m)] for i in range(m)] # weight matrix
    W = [[0 for j in range(m)] for i in range(m)] # weight matrix
    
    "Generating 2D Grid"
    G = grid_2d(row,column,periodic=False,create_using=None)
    
#     "Adding extra edges to get 2D Torus"
#     edges_rows = [((0,0),(0,4)),((1,0),(1,4)),((2,0),(2,4)),((3,0),(3,4))] 
#     column_rows = [((0,0),(3,0)),((0,1),(3,1)),((0,2),(3,2)),((0,3),(3,3)),((0,4),(3,4))] 
#     G.add_edges_from(edges_rows)
#     G.add_edges_from(column_rows)
    nx.draw_networkx(G)
    plt.savefig('2D-Grid')
    plt.axis('off')
    plt.show()
    
    "Changing edges format to 1D so that it becomes easy to access the indices"
    
    edges = []
    for i in range(row):
        for j in range(column-1):
            edges.append([j+i*column,j+1+i*column])

    for i in range(row-1):
        for j in range(column):
            edges.append([j+i*column,j+(i+1)*column])      

    "Adding extra edges to get 2D Torus from 2D grid"

    for i in range(column):
        edges.append([i , i + (row-1)*column])

    for i in range(row):
        edges.append([i*column , row + i*column]) 

    print ('edges:',edges)   
    print ('total edges in 2D Torus:',len(edges))
    
    for [u, v] in edges:
        W[u][v] = rd.uniform(w1,w2)
        W[v][u] = rd.uniform(w1,w2)

    for i in range(m):
#         W[i][i] = w1+3*w2
        W[i][i] = m*w2
       
    for i in range(m):
        s1 = sum(W[i])
        W[i] = [W[i][j]/s1 for j in range(m)]       
    for [u, v] in edges:
        A[u][v] = min(W[u][v], W[v][u])
        A[v][u] = min(W[u][v], W[v][u])
    
    for i in range(m):
        A[i][i] = 1 - sum(A[i])

    
    B = np.matrix(A)
    print ((B.transpose() == A).all()) ## check symmetric property of A. It will print True
    print ([sum(A[i]) for i in range(m)])
    return A


# In[ ]:


"""
consensus algorithm, W is the weight matrix obtained from connected graph.
This function returns epsilon approximation of the average of x_1, x_2, ...,x_m where x = [x_1,...,x_m]
"""

def acce_consensus(W,m,x,eta1,tau1):
    
    v = [[0 for i in range(len(x[0]))] for j in range(m)]

    x_new1 = x
    x_old1 = x

    for t in range(int(tau1)):
        x_old2 = list(x_old1)   ## z_k,t-1
        x_old1 = list(x_new1)   ## z_k,t
#         print ('z_k,t-1',x_old2[1][0])
#         print ('z_k,t',x_old1[1][0])
        for i in range(m):
            v1 = [W[i][j]*np.array(x_old1[j]) for j in range(m)]
            v[i] = [sum([v1[l][j] for l in range(m)]) for j in range(len(x[0]))] 
        first_term = (1+eta1)*np.array(v)
        sec_term = eta1*np.array(x_old2)
        x_new1 = np.subtract(first_term, sec_term)  ## z_{k,t+1}
#         print ('z_k,t+1',x_new1[1][0])
    return x_new1



# In[ ]:


"""
This function returns the weighted sum of x_1, x_2, ...,x_m where x = [x_1,...,x_m]
"""

def oneConsensus(W2,m,x):
    v = [[0 for i in range(len(x[0]))] for j in range(m)]
    for i in range(m):
        u = [W2[i][j]*np.array(x[j]) for j in range(m)]
        v[i] = [sum([u[q][j] for q in range(m)]) for j in range(len(x[i]))] 
        
    return v


# In[ ]:


"""
Momentum coefficient
"""

def update_t(k,alpha):
    a1 = k
    a2 = k + alpha 
    return a1/a2


# In[ ]:


"""
Gradient of logistic loss function at point x
"""

def logistic_grad(bFeature,bLabel,x,d1): ## d1 is the total number of samples
    summ = np.zeros(len(bFeature[0]))
#     cof = 2*lambdaa
#     d1 = len(X)
    a = len(bFeature) ## batch size
    for i in range(a):
        inp1 = np.dot(x,bFeature[i])
        n1 = (-bLabel[i])*np.array(bFeature[i])
        dom = bLabel[i]*(inp1)
        if (dom > 0): ## To avoid math flow error
            num1 = 1 - 1/(1 + math.exp((-1)*dom))
            #print (num1)
        else:
            num1 = 1/(1 + math.exp(dom))
        num2 = num1*np.array(n1) 
        summ = np.add(summ,num2)
    loss_der = (1/d1)*np.array(summ)
    return loss_der


# In[ ]:


"""
logistic function value computed at point x
"""

def logistic_func(X,Y,x): 
    """
    logistic function value computed at x
    Input:
    Features: X
    Labels: Y
    regcoef: lambda
    x: Point at which function value is computed

    Returns:
    Logistic regression function value at x with dataset (X,Y)
    """
    f = 0
    d1 = len(X)
#     xnorm = (LA.norm(x))**2
    for i in range(d1):
        inp = np.dot(x,X[i])
        dom = Y[i]*inp
        if (dom > 0):
            f1 = 1+ math.exp((-1)*dom)
            z11 = math.log(f1)
            f = f + z11   
            #print (z11)
        else:
            f1 = 1+ math.exp(dom)
            z11 = (-1)*dom + math.log(f1)
            f = f + z11   
            #print (z11)
            
         
    func = f/(d1)

    return func


# In[ ]:


"""
Projection of point v onto l2 ball of radius R centred at v0, if v lies outside the ball 
then Proj(v) = v0 + (R/||v-v0||)(v-v0)
"""

def projection_l2ball_centre_v0(v,v0,R):
    diff = np.subtract(v,v0)
    norm = LA.norm(diff)
#     print (norm)
    if (norm <= R):
#         print ('norm of projection point:',norm)
        return v
    else:
        scaling = R/norm
        scaled_vector = scaling*np.array(diff)
        projection = np.add(v0,scaled_vector)
#         print ('norm of projection point:',LA.norm(projection))

        return projection


# In[ ]:


"""

This function runs DAG (one step communication) starting from x0. It returns the average of local iterates and last iterate of all nodes. 

"""

def DAGOSC(x0,W,m,R,InIt,dataDis,step,epsilon_prevIte): ## d1 = total samples
   
    """
    x0: initial iterates x_{k,0}
    W: weight matrix
    m: number of nodes
    InIt: number of inner iterates t_k
    dataDis: local data sets
    step: step size s_k
    r: diameter of constraint set X
    epsilon_prevIte: consensus error ub epsilon_k
    """
    
    x = np.zeros((nodes,len(X[0])))
    func = np.zeros(InIt) # function values
    x_wp = np.copy(x)
    z1 = np.copy(x0)
    y = np.copy(x0)
    for t in range(InIt):
        v3 = oneConsensus(W,m,y)
        for i in range(m):
            feature = dataDis[0][i] ## ith block of data
            label = dataDis[1][i]
            grad = logistic_grad(feature,label,y[i],d1)
            step_grad = step*np.array(grad)
            #norm_gradf[k] = norm_gradf[k] + (LA.norm(grad))**2    # norm of gradient
            x_wp[i] = np.subtract(v3[i],step_grad)
            radius_new = R + epsilon_prevIte ### R_k + epsilon(k)
            x[i] = projection_l2ball_centre_v0(x_wp[i],x0[i],radius_new)
            
        if (t == 0):
            y = x.copy()
#             print ('y',y)
        else:
            mom1 = np.subtract(x,z1)
#             print ('mom1',mom1[0])
            beta = update_t(t,alpha)
#             print ('beta',beta)
            mom2 = beta*np.array(mom1)
#             print ('mom2',mom2[0])
            y = np.add(x,mom2)
#             print ('y',y[0])
        z1 = x.copy() ## store x^i of previous iterate
        avg_estimates = np.mean(x,axis = 0)
        func[t] = logistic_func(X,Y,avg_estimates)
        function_mdaca.write(str(func[t])+'\n')
        function_mdaca.flush()
    
    return x  ## local iterates


# In[ ]:


" inner iterations "
def inner_ite(Rk,epsilonk,step_size,alpha,L,varepsilon0,k):
    num1 = (alpha-1)*(Rk + epsilonk )*(2**(k/2))
    den1 = math.sqrt(2*step_size)
    num2 = 3*math.sqrt(2*L)*num1
    den2 = math.sqrt(st ep_size*varepsilon0)
    t1 = math.ceil(num1/den1)
    t2 = math.ceil(num2/den2)
    print ('first term in tk',t1)
    print ('second term in tk',t2)
    innite = max(t1,t2)
#     print (type(innite))
    print ('int(innite)',innite)
    return innite


# In[ ]:


def eps_esptilde_vareps(L,old_vareps,B,k):
    new_eps = math.sqrt(old_vareps/(18*L**2))
    new_epstilde = math.sqrt(old_vareps/18)
    new_vareps = old_vareps/2
    den = 1+L*(2**k -2)
    new_omega = new_vareps + (6*L**2*B)/den + (new_vareps*L)/(3*den)

    return new_eps,new_epstilde,new_vareps,new_omega


# In[ ]:


def tau_iterates(nodes,Rk,old_eps,new_eps,muW):
    
    num1 = math.log(2*math.sqrt(nodes)*(Rk + 2*old_eps)) - math.log(new_eps)
    den1 = (-1)*math.log(1-math.sqrt(1-muW))
    tauk = math.ceil(num1/den1)
    
    return tauk


# In[ ]:


def tau_gradients(L,x0,e0,nodes,old_eps,new_epstilde,muW):
    
    objx0 = logistic_func(X,Y,x0)
    fterm = 4*nodes*L*(math.exp(1)*e0 + (math.exp(0.5)+1)*objx0 + (4/3)*math.exp(0.5) +2)
    secterm = 4*nodes*L*objx0 + 2*nodes*(L**2)*(old_eps**2)
    ## computing upper bound of ||grad F(x_{k+1,0})||
    gradub = math.sqrt(fterm + secterm)
    
    num1 = math.log(2*gradub +2) - math.log(new_epstilde)
    den1 = (-1)*math.log(1-math.sqrt(1-muW))
    
    tautilde = math.ceil(num1/den1)
    return tautilde
     


# In[ ]:


def stepsize(L,nodes,B,vareps0,lambdaa_minW,muW,k):
    
    fterm = lambdaa_minW/L
#     maxfterm = (72*nodes*B*(L**2))/vareps0

#     den1 = 2*(max(maxfterm,nodes*L)*(2**k) + 2*nodes*L + L)
    den1 = 0.0001+L*(2**k)
    secterm = (1-muW)/den1
    print ('fterm, secterm:',fterm,secterm)
    if (fterm >= secterm):
        return fterm/(2**k)
    else:
        return secterm


# In[ ]:


def mDACA(X,Y,T,nodes,W,d1,alpha,zeta,eta1,c_0):

#     z = [[0 for i in range(len(X[0]))]  for j in range(m)] 
    z = np.zeros((nodes,len(X[0])))
    initialx0 = np.zeros(len(X[0]))
#     func = np.zeros(T+1) # function values
    dataDis = data_blocks(X,Y,nodes)
    func = logistic_func(X,Y,z[0])
    function_mdaca.write(str(func)+'\n')
    function_mdaca.flush()
    tauIt = np.zeros(T) ## stores number of communications in consensus on iterates
    tauGrad = np.zeros(T) ## stores number of communications in consensus on gradients
    c_val = np.zeros(T) ## stores c_k at every outer iterate
    innIter = np.zeros(T)
    cycles = np.ones(T) ## stores number of cycles at every outer iterate k
    xCons = np.copy(z)
    xCons_old = np.copy(z) ## consensus iterates at previous outer iterate
    "Initialize varpsilon_0"
    grad_0 = logistic_grad(X,Y,xCons[0],d1)
    vareps0 = 4 + 4*(LA.norm(grad_0))**2 + 10**7
    print ('varepsilon0',vareps0)
    new_eps = 1 ## epsilon_1
    new_epstilde = 1 ## epsilontilde_1
    new_vareps = vareps0/2 ## varepsilon_1 
    new_omega = vareps0/2
    c = c_0
    B = func ## f(x0) because f^lb = 0 for logistic loss
    e0 = 2*B ## choosing e0 = 2f(x0)
    for k in range(T):
        """
        Outer iteration
        """
        step = stepsize(L,nodes,B,vareps0,lambdaa_minW,muW,k)
#         if (k <= 20): ## using large step size in the beginning
#         step = lambdaa_minW/(L*2**k)
        print ('step size at k=',k+1,'is',step)
        ## store old epsilons before updating them
        old_eps = np.copy(new_eps)
        old_epstilde = np.copy(new_epstilde)
        old_vareps = np.copy(new_vareps)
        old_omega = np.copy(new_omega)
        
        ## compute new epsilon, epsilon tilde, varepsilon
        new_eps,new_epstilde,new_vareps,new_omega = eps_esptilde_vareps(L,old_vareps,B,k+1)
        ## compute radius Rk
        Rk = (c**2)*(math.sqrt(old_omega)+old_epstilde+L*old_eps)
        print ('radius Rk at k=',k+1,'is',Rk)
#         print (tauIt[k])
#         print (tauGrad[k])
        ## compute inner iterations tk
        inIt = inner_ite(Rk,old_eps,step,alpha,L,vareps0,k+1)
        innIter[k] = inIt
        print ('inner iterations at k =',k+1,'are',innIter[k])

#         InIt = 10**4
        inner_iterates_cycles.write(str(innIter[k])+'\n')
        inner_iterates_cycles.flush()
#         if (k <= 10): ## using large step size in the beginning
#             step = lambdaa_minW/L
#         print ('step size:',step)
#         np.savetxt('innIter_outItek_check.txt',innIter) ## store inner iterations for every outer iteration k
        z = DAGOSC(xCons_old,W,nodes,Rk,inIt,dataDis,step,old_eps)
    
        ## compute tau_k+1 and tau_tilde_k+1
        tauIt[k] = tau_iterates(nodes,Rk,old_eps,new_eps,muW)
        num_comm_iterates.write(str(tauIt[k])+'\n')
        num_comm_iterates.flush()
        xCons = acce_consensus(W,nodes,z,eta1,tauIt[k])
        
        grad_est = [] ## stores grad f_i at consensus iterate xCons^i
        for i in range(nodes):
            feature = dataDis[0][i] ## ith block of data
            label = dataDis[1][i]
            grad_est.append(logistic_grad(feature,label,xCons[i],d1))
        
        tauGrad[k] = tau_gradients(L,initialx0,e0,nodes,old_eps,new_epstilde,muW)  ## tilde{tau}_k
        grad_cons = acce_consensus(W,nodes,grad_est,eta1,tauGrad[k])  ## cal F_{k,0}
        num_comm_gradients.write(str(tauGrad[k])+'\n')
        num_comm_gradients.flush()
        
#         varepsln = varepsilon(g1,g2,g3,varepsilon_prevIte,c,step,k)
        print ('varepsilon at previous iterate',old_vareps)
#         print ('varepsilon at k =',k+1,'is',new_vareps)
        normGradC = [(LA.norm(grad_cons[i]))**2 for i in range(nodes)]
#         print ('norm of cal Fi_s at k = ',k+1,'are',normGradC)
        check = [normGradC[i] <= new_vareps for i in range(nodes) ]
        print ('gradient norm at all nodes (cal F)',normGradC)
        print ('varepsilon(k+1)',new_vareps)
        while (any(check) == False): ## any(check) returns True if atleast one is True in check array otherwie returns False
            cycles[k] += 1
            c = zeta*c
            Rk = (c**2)*(math.sqrt(old_omega)+old_epstilde+L*old_eps)
    #         print (tauIt[k])
    #         print (tauGrad[k])
            ## compute inner iterations tk
            inIt = inner_ite(Rk,old_eps,step,alpha,L,vareps0,k+1)
            innIter[k] = inIt
            print ('inner iterations at k =',k+1,'are',innIter[k])

    #         InIt = 10**4
            inner_iterates_cycles.write(str(innIter[k])+'\n')
            inner_iterates_cycles.flush()

    #         np.savetxt('innIter_outItek_check.txt',innIter) ## store inner iterations for every outer iteration k
            z = DAGOSC(xCons_old,W,nodes,Rk,inIt,dataDis,step,old_eps)

            ## compute tau_k+1 and tau_tilde_k+1
            tauIt[k] = tau_iterates(nodes,Rk,old_eps,new_eps,muW)
            num_comm_iterates.write(str(tauIt[k])+'\n')
            num_comm_iterates.flush()
            xCons = acce_consensus(W,nodes,z,eta1,tauIt[k])

            grad_est = [] ## stores grad f_i at consensus iterate xCons^i
            for i in range(nodes):
                feature = dataDis[0][i] ## ith block of data
                label = dataDis[1][i]
                grad_est.append(logistic_grad(feature,label,xCons[i],d1))

            tauGrad[k] = tau_gradients(L,initialx0,e0,nodes,old_eps,new_epstilde,muW)  ## tilde{tau}_k
            grad_cons = acce_consensus(W,nodes,grad_est,eta1,tauGrad[k])  ## cal F_{k,0}
            num_comm_gradients.write(str(tauGrad[k])+'\n')
            num_comm_gradients.flush()

    #         varepsln = varepsilon(g1,g2,g3,varepsilon_prevIte,c,step,k)
            print ('varepsilon at previous iterate',old_vareps)
            print ('varepsilon at k =',k+1,'is',new_vareps)
            normGradC = [(LA.norm(grad_cons[i]))**2 for i in range(nodes)]
    #         print ('norm of cal Fi_s at k = ',k+1,'are',normGradC)
            check = [normGradC[i] <= new_vareps for i in range(nodes) ]
        
        
        xCons_old = np.copy(xCons)  
        c_val[k] = c
        c_values_OuterIte.write(str(c_val[k])+'\n')
        c_values_OuterIte.flush()
        print ('outer iteration' ,k+1,'done')
                            
                 
    return True
                            


# In[ ]:


## Computing Lipschitz parameter

X3 = np.matrix(X)

M1 = np.matmul(np.transpose(X3),X3)
M = M1/len(X3)
from numpy.linalg import eig
values , vectors = eig(M)
print (max(values)) ## L is less than equal to maximum eigen value

L = float(max(values))
print (L)


# In[ ]:


"creating files to store output"

function_mdaca = open(r"func_mDACA_a4a_2dTorus.txt","w")
inner_iterates_cycles = open(r"inner_iterates_involving_in_cycles.txt","w")
num_comm_iterates = open(r"num_comm_iterates.txt","w")
num_comm_gradients = open(r"num_comm_gradients.txt","w")
c_values_OuterIte = open(r"c_values.txt","w")


# In[ ]:


"Initializations"

row = 4 ### total number of rows in 2D Torus
column = 5 ### column = row+1, total number of columns in 2D Torus
nodes = row*column ### total number of nodes

d1 = len(X) ## total number of samples 
T = 500 ## number of outer iteartions
alpha = 10
c_0 = 2
zeta = math.sqrt(2)


# In[ ]:


"Generate weight matrix W and compute its eigenvalues"

rd.seed(100)
W = gen_graph(row,column,nodes,0.09,0.2)

np.savetxt('weight_mat.txt',W)
from numpy.linalg import eig
values , vectors = eig(W)
values = np.sort(values)
print('eigen values',values)
muW = values[nodes-2] ## second largest eigenvalue of W
print ('second largest eigenvalue of W:',muW)
lambdaa_minW = values[0] ## smallest eigenvalue of W
print ('smallest eigenvalue of W:',lambdaa_minW)

eta1 = (1 - math.sqrt(1 - muW**2))/(1 + math.sqrt(1 - muW**2))


# In[ ]:


## Create data partition

[BX,BY] = data_blocks(X,Y,nodes)
for i in range(nodes):
    minus1 = BY[i].count(-1)
    one  =  BY[i].count(1)

minus1 = Y.count(-1)
one = Y.count(1)

print (len(X))


# In[ ]:


func = mDACA(X,Y,T,nodes,W,d1,alpha,zeta,eta1,c_0)


# In[ ]:




