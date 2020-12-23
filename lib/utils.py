import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
import numpy as np
import os
import glob
import gc
from math import *
import pickle

def f(x):
    '''
    Returns factorial of argument.
    Example: f(3) returns 6.
    '''
    
    f = factorial(x)
    return f

def calc_coef(x, N, N1, N2, A, dz):
    '''
    Generates z airfoil coordinate from A coeffs of top OR bottom airfoil surface.
    -----------
    Inputs:
    x              # Numpy array of x coordinates
    N              # Qty of A coefs for surface    
    N1             # nose form
    N2             # nose form
    A              # Numpy array of A coeffs for surface
    dz             # trailing edge thickness
    -----------
    Outputs:
    z[len(x)]      # Numpy array with surface z coords
    '''
    
    z = np.zeros_like(x)
    
    for zc in range(len(z)):
        summa=0
        
        for i in range(N):
            summa += A[i]*(f(N)/(f(i)*f(N-i)))*(x[zc]**i)*((1-x[zc])**(N-i))        
            
        z[zc] = (x[zc]**N1)*((1-x[zc])**N2)*summa+x[zc]*dz
        
    return z

def get_airfoil_coords(Au, Al, N1=0.5, N2=1.0, n_coords=100, dz=0.):
    '''
    Generates arrays of absolute x-z airfoil coordinates from A coeffs of top and bottom airfoil surface.
    -----------
    Inputs:
    Au, Al         # Numpy arrays of A coeffs 
    N1 = 0.5       # nose form
    N2 = 1         # nose form
    n_coords = 100 # quantity of x absolute coords
    dz = 0.        # trailing edge thickness
    -----------
    Outputs:
    x[n_coords], z[n_coords] - Numpy arrays with foil coords
    '''
    
    Nu = len(Au)
    Nl = len(Al)
    
    # Create x coordinate
    x=np.ones((n_coords))   
    zeta=np.zeros((n_coords));
    for i in range(1, n_coords+1):
        zeta[i-1]=2*pi/n_coords*(i-1);
        x[i-1]=0.5*(cos(zeta[i-1])+1);
    zeroind = np.argmin(x)
    
    xl= np.hstack([x[:zeroind],np.array([0])]) 
    xu = np.hstack([x[zeroind:],np.array([1])]) 
    x = np.hstack([xl,xu])

    
    zu = calc_coef(xu, Nu, N1, N2, Au, dz)
    zl = calc_coef(xl, Nl, N1, N2, Al, -dz)
    
    z = np.hstack([zl,zu])    
    
    return x,z

def array_from_coefs(Au, Al, N1=0.5, N2=1.0, n_coords=100, dz=0.0011111111111111111):
    '''
    Generates 512x512 bool array/bitmap of airfoil from A coeffs of top and bottom airfoil surface.
    -----------
    Inputs:
    Au, Al         # Numpy arrays of A coeffs 
    N1 = 0.5       # nose form
    N2 = 1         # nose form
    n_coords = 100 # quantity of x absolute coords
    dz = 0.        # trailing edge thickness
    -----------
    Outputs:
    512x512 bool Numpy array
    '''
    
    x, z = get_airfoil_coords(Au, Al, N1=0.5, N2=1.0, n_coords=100, dz=0.0011111111111111111)
    fig = plt.figure(figsize=(7.12,7.12),  frameon=False)
    canvas = FigureCanvasAgg(fig)
    ax = fig.add_axes([0.,0.,1.,1.])
    ax.fill(x,z, 'black')
    ax.axis('equal');
    ax.axis('off');
    s, (width, height) = canvas.print_to_buffer()
    foil_bmp = np.array(np.frombuffer(s, np.uint8).reshape((height, width, 4))[:,:,0])
    plt.close()
    foil_bmp[foil_bmp==0] = 1
    foil_bmp[foil_bmp==255] = 0
    foil_bmp = foil_bmp.astype('bool') 
    return foil_bmp

def deform(spline, span_position=80, width=10, depth=10, positive=True):
    '''
    Generates arbitrary deformation of airfoil surface by updating A coefs. All variables in percents relatively to airfoil chord!
    ---------
    Inputs:
    spline         # A coefs array
    span_position  # relative position of deformation center
    width          # relative width of deformation
    positive       # deform spline up if True and down if False
    ---------
    Outputs:
    new_spline     # A coefs array of deformed spline

    
    '''
    span_position/=100
    width/=100
    depth/=100
    
    if len(spline.shape)!=1:
        raise Exception("Wrong input array!")
    
    n_points = len(spline)    
    midpoint = int(round(n_points*span_position, 0))    
    leftpoint = midpoint - int(round(n_points*width, 0))
    rightpoint = midpoint + int(round(n_points*width, 0))
    if leftpoint<0:
        leftpoint=0
    if rightpoint>n_points:
        rightpoint=n_points
        
    coefs = np.ones(rightpoint-leftpoint+1)    
    n_steps = int(len(coefs)/2)+1    
    step = depth/(n_steps-1)
    
    if not positive:    
        step = -step
        
    for i in range(0, n_steps-1):
        coefs[i+1]  = coefs[i] + step
        coefs[-i-2] = coefs[-i-1] + step
    coefs[i] = coefs[i-1] + step

    new_spline = spline.copy()
    coef_pos = 0
    for i in range(leftpoint, rightpoint+1, 1):
        new_spline[i-1] = new_spline[i-1]*coefs[coef_pos]
        coef_pos+=1

    return new_spline

def save_pkl(data, filename):
    '''
    Saves any object as pickle file. Filename shall contain full path.
    '''
    with open(filename, 'wb') as output:
        pickle.dump(data, output)
        
def load_pkl(filename):
    '''
    Loads object from pickle file. Filename shall contain full path.
    '''
    with open(filename, 'rb') as input:
        return pickle.load(input)
