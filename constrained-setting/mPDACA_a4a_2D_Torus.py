#!/usr/bin/env python
# coding: utf-8

# # Import required packages

# In[ ]:


from matplotlib.collections import EventCollection
import numpy as np
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
Calling Condat's Fast projection to L1 ball 
Reference: Condat L., Fast projection onto the simplex and the ℓ1 ball. Mathematical
Programming, 2016
"""

get_ipython().run_line_magic('run', './Fast_projection_simplex_l1ball.ipynb')

"""
Calling Dykstra algorithm to project on the intersection of L1 and L2 ball
Reference: Birgin, E.G., Raydan, M.: Robust stopping criteria for dykstra’s algorithm.
SIAM Journal on Scientific Computing, 2005
"""

get_ipython().run_line_magic('run', './Dykstra_projection_L1_ball.ipynb')


# In[ ]:


"""
Download data set a4a from LIBSVM
Website: https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html
Load csv file of downloaded data 
"""

mydata = np.genfromtxt('a4a3.csv',delimiter = ',')
num_features = len(mydata[0]) - 1
num_samples = len(mydata)
print('num samples: %d num features : %d ' %(num_samples, num_features))


# In[ ]:


"""
Shuffling the indices
"""

indices = np.arange(len(mydata))
np.random.seed(100)
np.random.shuffle(indices)
print ('indices after shuffling',indices)
shuffled_data = [list(mydata[i]) for i in indices]


"""
Separating features and labels from shuffled data
"""

shuffled_data = np.array(shuffled_data)
X1 = shuffled_data[:,1:(num_features + 1)]
Y1 = shuffled_data[:,:1]
print (len(X1[0]))
X = [X1[i]/LA.norm(X1[i]) for i in range(len(X1))] ## Normalize feature vectors
Y = [Y1[i][0] for i in range(len(Y1)) ]

"""
Adding 1 to each feature vector to incorporate offset parameter
"""

X = [ np.insert(X[i],0,1) for i in range(len(X))]


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
    d = len(X)
    s2 = int(d/m)  ## size of each block
    g = 0
    for i in range(m):
        for j in range(g,s2 + g,1):
            BX[i].append(X[j])
            BY[i].append(Y[j])
        g = g + s2 
    samples_after_eq_dist = m*int(d/m)  ## total samples after equally distributed samples to every node
    remaining_samples = d - m*int(d/m) ### This quantity will always be less than equal to m
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
    w1, w2: weights to generate entries of W 
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
        W[i][i] = w1+3*w2
       
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


def multiConsensus(W,m,x,tau):
    
    """
    W: weight matrix obtained from connected graph
    tau: number of communication rounds
    Output: This function returns approximation of the average of x_1, x_2, ...,x_m
    where x = [x_1,...,x_m]
    """
    
    v = [[0 for i in range(len(x[0]))] for j in range(m)]
    x_new = np.copy(x)
    for t in range(int(tau)):
        x_old = np.copy(x_new)
        for i in range(m):
            v2 = [W[i][j]*np.array(x_old[j]) for j in range(m)]
            v2 = np.array(v2)
            v[i] = v2.sum(axis =0)
        x_new = np.copy(v)     
    return x_new


# In[ ]:


def oneConsensus(W2,m,x):
    """
    This function returns the weighted sum of x_1, x_2, ...,x_m where x = [x_1,...,x_m]
    """
    v = [[0 for i in range(len(x[0]))] for j in range(m)]
    for i in range(m):
        u = [W2[i][j]*np.array(x[j]) for j in range(m)]
        u = np.array(u)
        v[i] = u.sum(axis = 0)
        
    return v


# In[ ]:


"""
Momentum coefficient (t-1)/(t+alpha-1)
"""

def update_t(k,alpha):
    a1 = k
    a2 = k + alpha 
    return a1/a2


# In[ ]:


"""
Gradient of logistic loss function at point x
"""

def logistic_grad(bFeature,bLabel,x,lambdaa,d): ## d is the total number of samples
    summ = np.zeros(len(bFeature[0]))
    cof = (lambdaa)/m
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
    loss_der = (1/d)*np.array(summ)
    reg_der = cof*np.array(x)
    gradient = np.add(loss_der,reg_der)
    return gradient


# # Finding logistic regression function value at point x

# In[ ]:


def logistic_func(X,Y,x,lambdaa): 
    """
    logistic function value computed at (x,y)
    Input:
    Features: X
    Labels: Y
    regcoef: lambda
    x: Point at which function value is computed

    Returns:
    Logistic regression function value at x with dataset (X,Y)
    """
    
    f = 0
    d = len(X)
    cof = (lambdaa)/2
    xnorm = (LA.norm(x))**2
    for i in range(d):
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
         
    func = (f/d) + cof*xnorm

    return func


# In[ ]:


"""

This function runs projected distributed accelerated gradient method (one step communication) starting from x0. 
It returns the last local iterates , r is the diameter of the orginal constraint set X, R = R_k 
"""

def DAGOSC(x0,W,m,InIt,dataDis,step,r,R,epsilon_prevIte,tolerance,d): 
    
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
    
    x = [[0 for i in range(len(X[0]))]  for j in range(m)] ## store iterates x_k of all agents at current iterate
    x_p = [[0 for i in range(len(X[0]))]  for j in range(m)] ## projected iterate
    func = np.zeros(InIt) # function values
    z1 = x0.copy()
    y = x0.copy()
    for t in range(InIt):
        v3 = oneConsensus(W,m,y)
        for i in range(m):
            feature = dataDis[0][i] ## ith block of data
            label = dataDis[1][i]
            grad = logistic_grad(feature,label,y[i],lambdaa,d)
            step_grad = step*np.array(grad)
            x[i] = np.subtract(v3[i],step_grad)
            """
            project x_i onto the intersection of X and X^i_k
            """
#             print ('x[0]',x[0][0])
            radius_new = R + epsilon_prevIte ### R_k + epsilon(k)
            x_p[i] = Dikstra(x[i],x0[i],r,radius_new,tolerence)
#             print ('x_p[0]',x_p[0][0])
#         print ('inner iteration',t+1)
        if (t == 0):
            y = x_p.copy()
        else:
            mom1 = np.subtract(x_p,z1)
            beta = update_t(t,alpha)
            mom2 = beta*np.array(mom1)
            y = np.add(x_p,mom2)
        z1 = x_p.copy() ## store x^i of previous iterate
        avg_estimates = np.mean(x_p,axis = 0)
        func[t] = logistic_func(X,Y,avg_estimates,lambdaa)
        function_mpdaca.write(str(func[t])+'\n')
        function_mpdaca.flush()    
    return x_p  ## projected local iterates


# In[ ]:


" inner iterations "
def innIt(c,m,L,mu,B,varepsilon_0,k):
    s0 = (m*L*(1-mu)*varepsilon_0)/(16*B**2)
    t1 = (4*(alpha -1)*math.sqrt(m))/(math.sqrt(L*s0))
    t2 = 1.5*(1+L*c**2) + 1/16
#     print (t1,t2,t3,t4)
    return t1*t2*(math.sqrt(2)**k)

"""
updating varepsilon, epsilon, epsilon_tilde
"""

def varepsilon_k(varepsilon_prevIte):
    varepsilon = varepsilon_prevIte/2
    epsilon = (math.sqrt(varepsilon))/16
    epsilon_tilde = (math.sqrt(varepsilon))/8
    return varepsilon, epsilon, epsilon_tilde

def tau_iterates(epsilon,m,mu,radius):
    den = (-1)*(math.log(mu))
    num = math.log(math.sqrt(2*m)*(radius+epsilon)) - math.log(epsilon)
    tau = num/den
    return math.ceil(tau)

def tau_tilde(epsilon_tilde,epsilon,mu,B,radius):
    C1 = m*epsilon + B/L
    den = (-1)*(math.log(mu))
    num = math.log(C1) - math.log(epsilon_tilde)
    tau_tilde = num/den
    return math.ceil(tau_tilde)
    


# In[ ]:


def mPDACA(X,Y,T,m,W,d,alpha,zeta,c_0,radius):
    norm_proximal_grad = np.zeros(T+1) ## stores maximum local proximal gradient square norm
    z = [[0 for i in range(len(X[0]))]  for j in range(m)] 
    func = logistic_func(X,Y,z[0],lambdaa)
    function_mpdaca.write(str(func)+'\n')
    function_mpdaca.flush()
    tauIt = np.zeros(T) ## tau = stores number of communications in consensus on iterates
    tauGrad_step = np.zeros(T) ## tau_tilde = stores number of communications in consensus on gradient descent step
    c_val = np.zeros(T) ## stores c_k at every outer iterate
    innIter = np.zeros(T)
    cycles = np.ones(T) ## stores number of cycles at every outer iterate k
    xCons = np.copy(z)
    xCons_old = np.copy(z) ## consensus iterates at previous outer iterate
    
    "Initialize varpsilon_0"
    dataDis = data_blocks(X,Y,m)
    sum_norm_grad_f_i = 0 ### sum of square norm of  \nabla f_i(x^i_0)
    grad_f_i = [] ### store local gradients at initial points x^i_0
    for i in range(m):
        bFeature = dataDis[0][i] ## ith block of data
        bLabel = dataDis[1][i]
        grad = logistic_grad(bFeature,bLabel,z[0],lambdaa,d)
        step_grad = (-1/L)*(np.array(grad))
        grad_f_i.append(step_grad)
        grad_norm = (LA.norm(grad))**2
        sum_norm_grad_f_i = sum_norm_grad_f_i + grad_norm
    norm_v1_0 = (1/L)*math.sqrt(sum_norm_grad_f_i) ### since x_0 = z = 0
    "tilde{tau}_1"
    numerator = math.log(norm_v1_0*(10**10)) ### take epsilon_tilde(1) = 10^-10
    denominator = (-1)*math.log(mu)
    tau_tilde0 = numerator/denominator
    v1_tilde = multiConsensus(W,m,grad_f_i,int(tau_tilde0))  ### \tilde{v_1,0}
    project_v1_tilde = projection_l1_ball(v1_tilde[0], radius)  ## choose i0 = 0 (node 1) 
    q_1_square_norm = LA.norm(project_v1_tilde)**2
    norm_proximal_grad[0] = q_1_square_norm
    proximal_grad_norm.write(str(norm_proximal_grad[0])+'\n')
    proximal_grad_norm.flush()
    varepsilon_0 = 0.1
    varepsilon_prevIte = varepsilon_0/2 ## varepsilon_1
    
    "Initialize s^'_0"
    
    step = (m*varepsilon_0*L*(1-mu))/(16*(B**2))
#     step = (s0*accuracy)/varepsilon_0
    "Initialize R_1 and epsilon_1"
    
    epsilon_prevIte = 0 ## because all x^i's are intialized to same vector 0 and hence epsilon(1) = 0
    epsilon_tilde_prevIte = 10**(-10) ## epsilon_tilde(1) = = 10^-10 
    c = c_0
#     R = 2*(1+(c**2)*m*L)*(math.sqrt(varepsilon_prevIte) + 2*epsilon + epsilon_tilde)
    
    """
    Outer iteration
    """
    for k in range(T):
        step = step/2
        print ('step size at k =',k+1,'is',step)

        InIt = innIt(c,m,L,mu,B,varepsilon_0,k+1)
        InIt = int(InIt)
        innIter[k] = InIt
        print ('inner iterations at k =',k+1,'are',InIt)

#         InIt = 10**4
        inner_iterates_cycles.write(str(innIter[k])+'\n')
        inner_iterates_cycles.flush()
        R = 2*(1+(c**2)*L)*(math.sqrt(varepsilon_prevIte) + 2*epsilon_prevIte + epsilon_tilde_prevIte)
        print ('R:',R)
        print ('varepsilon:',varepsilon_prevIte)
        z = DAGOSC(xCons_old,W,m,InIt,dataDis,step,radius,R,epsilon_prevIte,tolerence,d)
        varepsilon, epsilon, epsilon_tilde = varepsilon_k(varepsilon_prevIte)
#         print ('varepsilon_prevIte after computing varepsilon',varepsilon_prevIte)

        tauIt[k] = tau_iterates(epsilon,m,mu,radius)
        num_comm_iterates.write(str(tauIt[k])+'\n')
        num_comm_iterates.flush()
        xCons = multiConsensus(W,m,z,tauIt[k])
        v_k = [[0 for i in range(len(X[0]))]  for j in range(m)]
        for i in range(m):
            feature = dataDis[0][i] ## ith block of data
            label = dataDis[1][i]
            local_grad = logistic_grad(feature,label,xCons[i],lambdaa,d)
            step_localGrad = (1/L)*np.array(local_grad)
            v_k[i] = np.subtract(xCons[i],step_localGrad)
        
        "Consensus on v_k+1 to compute v_tilde_k+1"
        
        tauGrad_step[k] = tau_tilde(epsilon_tilde,epsilon,mu,B,radius)        ## tilde{tau}_k
        num_comm_gradients.write(str(tauGrad_step[k])+'\n')
        num_comm_gradients.flush()
        v_tilde = multiConsensus(W,m,v_k,tauGrad_step[k])
        
        proximal_grad = [] ## stores q_i at consensus iterate xCons^i
        for i in range(m):
            proj = projection_l1_ball(v_tilde[i], radius)
            diff = np.subtract(xCons[i],proj)
            square_norm = (LA.norm(diff))**2
            proximal_grad.append(square_norm)
        
        check = [proximal_grad[i] <= varepsilon for i in range(m) ]
        while (any(check) == False): ## any(check) returns True if atleast one is True in check array otherwie returns False
            cycles[k] += 1
            np.savetxt('cycles.txt',cycles)
            c = zeta*c
            print ('c update',c)
            InIt = innIt(c,m,L,mu,B,varepsilon_0,k+1,accuracy)
            innIter[k] = InIt
            print ('inner iterations at k =',k+1,'are',InIt)

#             InIt = 10**4

            inner_iterates_cycles.write(str(innIter[k])+'\n')
            inner_iterates_cycles.flush()
            R = 2*(1+(c**2)*L)*(math.sqrt(varepsilon_prevIte) + 2*epsilon_prevIte + epsilon_tilde_prevIte)
            print ('R:',R)
            z = DAGOSC(xCons_old,W,m,InIt,dataDis,step,radius,R,epsilon_prevIte,tolerence,d)
#             tauIt[k] = tau_iterates(epsilon,m,r,mu)
#             num_comm_iterates.write(str(tauIt[k])+'\n')
#             num_comm_iterates.flush()
            xCons = multiConsensus(W,m,z,tauIt[k])
            v_k = [[0 for i in range(len(X[0]))]  for j in range(m)]
            for i in range(m):
                feature = dataDis[0][i] ## ith block of data
                label = dataDis[1][i]
                local_grad = logistic_grad(feature,label,xCons[i],lambdaa,d)
                step_localGrad = (1/L)*np.array(local_grad)
                v_k[i] = np.subtract(xCons[i],step_localGrad)

            "Consensus on v_k+1 to compute v_tilde"

#             tauGrad_step[k] = tau_tilde(epsilon_tilde,mu,B)        ## tilde{tau}_k
#             num_comm_gradients.write(str(tauGrad_step[k])+'\n')
#             num_comm_gradients.flush()
            v_tilde = multiConsensus(W,m,v_k,tauGrad_step[k])
            proximal_grad = [] ## stores q_i at consensus iterate xCons^i
            for i in range(m):
                proj = projection_l1_ball(v_tilde[i], radius)
                diff = np.subtract(xCons[i],proj)
                square_norm = (LA.norm(diff))**2
                proximal_grad.append(square_norm)

            check = [proximal_grad[i] <= varepsilon for i in range(m) ]
            
        "store minimum local proximal gradient squre norm"    
        norm_proximal_grad[k+1] = min(proximal_grad)
        proximal_grad_norm.write(str(norm_proximal_grad[k+1])+'\n')
        proximal_grad_norm.flush()
        xCons_old = np.copy(xCons)  
        c_val[k] = c
        c_values_OuterIte.write(str(c_val[k])+'\n')
        c_values_OuterIte.flush()
        varepsilon_prevIte = varepsilon
        epsilon_prevIte = epsilon
        epsilon_tilde_prevIte = epsilon_tilde 
        print ('outer iteration' ,k+1,'done')
                            
                 
    return True


# In[ ]:


"creating files to store data"

function_mpdaca = open(r"func_mPDACA_a4a3_2D_torus.txt","w")
proximal_grad_norm = open(r"minimum_local_proximal_grad_square_norm.txt","w")
inner_iterates_cycles = open(r"inner_iterates_involving_in_cycles.txt","w")
num_comm_iterates = open(r"num_comm_iterates.txt","w")
num_comm_gradients = open(r"num_comm_gradients.txt","w")
c_values_OuterIte = open(r"c_values.txt","w")


# In[ ]:


"""
Finding Lipschitz parameter L
"""
X3 = np.matrix(X)

M1 = np.matmul(np.transpose(X3),X3)
M = M1/len(X3)
from numpy.linalg import eig
values , vectors = eig(M)

L = float(max(values)) + lambdaa ## L is less than equal to (maximum eigen value of (1/d)X^TX) + lambd
print ('L:',L)


# In[ ]:


"Initializations"

tolerence = 10**(-20) ## tolerance on Dikstra algorithm
row = 4 ### total number of rows in 2D Torus
column = 5 ### total number of columns in 2D Torus
m = row*column ### total number of nodes
rd.seed(2)
W = gen_graph(row,column,m,0.04,0.05)
from numpy.linalg import eig
values , vectors = eig(W)
values = np.sort(values)
print('eigenvalues of W are',values)
mu = values[m-2] ## second largest eigen value of W
print ('\mu(W) = ',mu)
d = len(X) ## total number of samples 
T = 10**5 ## outer iterations
alpha = 3
c_0 = 0.1
radius = 1 ## X is l1 unit ball
zeta = math.sqrt(2)
lambdaa = 0.01 ## regularization parameter

## data distribution
[BX,BY] = data_blocks(X,Y,m)


"maximum samples a node has after distribution"

node_maximum_samples = max([len(BY[i]) for i in range(m)])

num_features = len(X[0])
B = (node_maximum_samples/d) + (lambdaa/m)*radius ## gradient bound or Lipschitz parameter of each f_i


# In[ ]:


func = mPDACA(X,Y,T,m,W,d,alpha,zeta,c_0,radius)

