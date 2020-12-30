from pathlib import Path
import pickle
import numpy as np
from scipy import interpolate
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from xfoil import XFoil
from xfoil.model import Airfoil
from config import *
from lib.utils import *



def show_foil_info(fpath, param_to_show='Cl'):
    '''
    Reads foil from pkl file. Requires FULL path to file.
    Shows thicknesses and desired foil curve before/after interpolations.
    '''

    param = {'Cl':0, 'Cd':1, 'Cm':2, 'Cp':3}

    foil = load_pkl(Path(fpath))
    fname = str(fpath).split("\\")[-1:]

    print("Foil name: %s \nRoot thickness: %2.2f, flap thickness: %2.2f" % (fname, foil['S'], foil['d']) )
    print("Show curve:", param_to_show)

    foil_array = foil['X']
    pre_foil_array = foil['X_raw']
    Re = foil_array[6, :, 0]
    alfas = foil_array[7, 0, :]

    for r in range(len(Re)):    
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 2))
        axes[0].set(title="Re="+str(Re[r])+" - after Xfoil")
        axes[1].set(title="Re="+str(Re[r])+" - after correction")
        axes[0].plot(alfas, pre_foil_array[param[param_to_show], r, :])
        axes[1].plot(alfas, foil_array[param[param_to_show], r, :])



def get_alfa_step(alfa_min, alfa_max, n_points_alfa):
    '''
    Finds and return proper alfa_step for xf.aseq to have exactly n_points_alfa alfas in xfoil predictions.
    '''
    x=np.array((1.        , 0.93442142, 0.86865613, 0.8027304 , 0.73666321,
               0.67046263, 0.6041302 , 0.53766349, 0.47106043, 0.40432225,
               0.33746473, 0.27055803, 0.20373882, 0.13734911, 0.07242776,
               0.01418394, 0.01418394, 0.07242776, 0.13734911, 0.20373882,
               0.27055803, 0.33746473, 0.40432225, 0.47106043, 0.53766349,
               0.6041302 , 0.67046263, 0.73666321, 0.8027304 , 0.86865613,
               0.93442142, 1.        ))
    
    y=np.array((0.00189   ,  0.01515836,  0.02746996,  0.0388957 ,  0.04947194,
                0.05917796,  0.06793897,  0.07562056,  0.08201013,  0.08680391,
                0.08957949,  0.08972539,  0.08637591,  0.07812218,  0.06213743,
                0.03019413, -0.03019413, -0.06213743, -0.07812218, -0.08637591,
               -0.08972539, -0.08957949, -0.08680391, -0.08201013, -0.07562056,
               -0.06793897, -0.05917796, -0.04947194, -0.0388957 , -0.02746996,
               -0.01515836, -0.00189   ))
    
    xf = XFoil()
    xf.airfoil = Airfoil(x, y)
    xf.max_iter = 16        
    xf.Re = 50000
    
    for rn in range(4):
    
        alfas = np.linspace(alfa_min, alfa_max, n_points_alfa+rn-1)
        
        alfa_step = alfas[1]-alfas[0]

        a, cl, cd, cm, cp = xf.aseq(alfa_min, alfa_max, alfa_step)
        
        if len(a)==n_points_alfa: return alfa_step, a
    
    raise Exception("Can not select right alfa step!")
    
    

def get_foil_flap_thickness(foil):
    '''
    Accept airfoil as xfoil Airfoil() object.
    Returns its flap thickness in 0...1 range.
    '''
    assert isinstance(foil, Airfoil), "This is not an xf.Airfoil object!"
    
    mid = int(foil.n_coords/2)
    x1 = mid-int(mid*flap_position)
    x2 = mid+int(mid*flap_position)

    thickness = abs(foil.y[x1]-foil.y[x2])
    
    assert thickness>0, "Foil_flap_thickness<=0!"
    
    return thickness



def get_foil_root_thickness(foil):
    '''
    Accept airfoil as xfoil Airfoil() object.
    Returns it main thickness in 0...1 range.
    '''
    assert isinstance(foil, Airfoil), "This is not an xf.Airfoil object!"
    
    thickness=0
    
    for pnt in range(int(foil.n_coords/2)):
        t = abs(foil.y[pnt]-foil.y[foil.n_coords-1-pnt])
        if t>thickness: thickness=t
            
    assert thickness>0, "Foil_root_thickness<=0!"
    assert thickness<1, "Foil_root_thickness is over 1!"
    
    return thickness



def read_airfoil_dat_file(fpath, silent=False):
    '''
    Reads file from absolute path fpath.
    Returns x and y arrays of coordinates.
    "silent" == do not show warnings. 
    '''
    
    assert isinstance(fpath, Path,), Exception('Parameter "%s" is not a Path instance' % fpath)
    
    file = open(fpath, 'r', newline='\r') 
    
    dat = file.read().split('\r')    
    
    x=[]
    y=[]

    for i in range(1, len(dat)-1):
        coords = dat[i].split()
        skipped_Xs = []
        if len(coords) == 2:
            try:
                x_c = coords[0]
                y_c = coords[1]                
                x_c = x_c.replace(")","").replace("(","")
                y_c = y_c.replace(")","").replace("(","")                
                try:
                    assert isinstance(float(x_c), float), str(fpath)
                    assert isinstance(float(y_c), float), str(fpath)
                    x.append(x_c)
                    y.append(y_c)
                except Exception as ex:
                    if not silent: print("W: Error in %s:" % (str(fpath) , str(ex)))
            except:
                if not silent: print("W: File '%s': can't read and convert X=%s and Y=%s, this coordinate skipped." 
                      % (str(fpath), str(coords[0]), str(coords[1])))                            
                skipped_Xs.append(x_c)
          
    if len(skipped_Xs)>0:
        assert (set(skipped_Xs) in x), Exception('E: Some coordinates are not imported in %s' % str(fpath))
        
    x = np.array(x).astype('float')
    y = np.array(y).astype('float')
    
    return x, y



def interpolate_airfoil(x, y, n_foil_points=128):
    '''
    Gets foil X and Y as numpy arrays of m points each.
    Returns resized arrays with n_foil_points each.    
    
    '''    
    assert isinstance(x, np.ndarray), "X is not an array"
    assert isinstance(y, np.ndarray), "Y is not an array"
    assert isinstance(n_foil_points, int), "n_foil_points is not int"
    assert(x.shape==y.shape), "X and Y are of different shapes"
    assert(n_foil_points>0), "n_foil_points can't be 0 or negative"
    
    interpolation_step = 1/(n_foil_points-1)
    
    tck, _ = interpolate.splprep([x, y], s=0)
    new_x = np.arange(0, 1+interpolation_step, interpolation_step)
    out = interpolate.splev(new_x, tck)
    
    return out[0], out[1]



def fill_gaps_in_xfoil_curve(curve, deg=3):
    '''    
    Takes curve X and Y arrays, interpolates middle NaNs, extrapolates head and tail NaNs.
    Returns arrays of the same size.    
    '''
    assert isinstance(curve, np.ndarray), "The curve is not an array"
    assert len(curve)>0, "The curve lenght is 0!"
    
    res_curve = curve.copy()
    
    x = np.arange(len(res_curve))
    has_values = ~np.isnan(res_curve)
    has_nans = np.isnan(res_curve)
    
    try:
        points = np.polyfit(x[has_values], res_curve[has_values], deg=3)
        res_curve[has_nans]  = np.polyval(points, x[has_nans])
    except Exception as ex:
        print(ex)
        print("x:", x)
        print("res_curve", res_curve)
    
    return res_curve


def create_foil_array_from_dat_file(fname, Re, alfas, alfa_min, alfa_max, alfa_step):
    
    # set up result dictionary
    foil_output = {}

    # set up foil data array
    foil_array = np.zeros((n_foil_params, n_points_Re, n_points_alfa))
    pre_foil_array = np.zeros((n_foil_params, n_points_Re, n_points_alfa))

    # load foil coords from file
    try:
        x, y = read_airfoil_dat_file(fname)
    except:
        raise Exception("W: Foil %s failed to read from file, skipped." % (fname))
        
    # keep in output dict
    foil_output['x_raw'] = x
    foil_output['y_raw'] = y

    # convert coords to n_foil_points
    try:
        x, y = interpolate_airfoil(x, y, n_foil_points)
    except:
        raise Exception("W: Foil %s failed to interpolate to %i points, skipped." % (fname, n_foil_points))        

    # set an Airfoil object on these coords
    current_foil = Airfoil(x,y)

    # get root and flap thicknesses 
    try:
        d = get_foil_flap_thickness(current_foil)
        S = get_foil_root_thickness(current_foil) 
    except:
        raise Exception("W: Foil %s failed to get thicknesses, skipped." % (fname))             

    # store in ary
    foil_array[4, :, :] = d
    foil_array[5, :, :] = S    

    # setup xfoil lib
    xf = XFoil()
    xf.airfoil = current_foil
    xf.max_iter = xfoil_max_iterations

    # xfoiling for each Re
    for num in range(len(Re)):

        xf.Re = Re[num]        
        a, cl, cd, cm, cp = xf.aseq(alfa_min, alfa_max, alfa_step)  

        assert (len(a)==n_points_alfa), "Lenght of alfa array is wrong!"

        nans_percent = sum(np.isnan(a))/len(a)

        # interpolate gaps in alfa plane
        if nans_percent < max_nans_in_curve:            
            cl = fill_gaps_in_xfoil_curve(cl)
            cd = fill_gaps_in_xfoil_curve(cd)
            cm = fill_gaps_in_xfoil_curve(cm)
            cp = fill_gaps_in_xfoil_curve(cp)

        # write results to main array
        foil_array[0, num, :] = cl
        foil_array[1, num, :] = cd
        foil_array[2, num, :] = cm
        foil_array[3, num, :] = cp
        foil_array[6, num, :] = Re[num]
        foil_array[7, num, :] = alfas

        # write results to debug array
        pre_foil_array[0, num, :] = cl
        pre_foil_array[1, num, :] = cd
        pre_foil_array[2, num, :] = cm
        pre_foil_array[3, num, :] = cp
        pre_foil_array[6, num, :] = Re[num]
        pre_foil_array[7, num, :] = alfas

        # interpolate gaps in Re plane
        for alfa in range(len(alfas)):
            foil_array[0, :, alfa] = fill_gaps_in_xfoil_curve(foil_array[0, :, alfa])
            foil_array[1, :, alfa] = fill_gaps_in_xfoil_curve(foil_array[1, :, alfa])
            foil_array[2, :, alfa] = fill_gaps_in_xfoil_curve(foil_array[2, :, alfa])
            foil_array[3, :, alfa] = fill_gaps_in_xfoil_curve(foil_array[3, :, alfa])            

    # save or discard data
    if np.sum(np.isnan(foil_array))==0:

        # fill output dict
        foil_output['X'] = foil_array
        foil_output['X_raw'] = pre_foil_array
        foil_output['y'] = current_foil
        foil_output['d'] = d
        foil_output['S'] = S

    else:
        
        raise Exception("W: Foil skipped - %i NaNs in foil_array." % int(np.sum(np.isnan(foil_array))))
        
    print("     --> %i NaNs corrected." % (int(np.sum(np.isnan(pre_foil_array)))))
        
    return foil_output

    

    