import numpy as np

from xfoil import XFoil
from xfoil.model import Airfoil

from lib.preprocess_modules import *
from config import *

def prepare_foil_to_predict(fname, n_layers=8):
    
    # get list of alfas and alfa step
    alfa_step, alfas = get_alfa_step(alfa_min, alfa_max, n_points_alfa)

    # set list of Re's
    Re = np.linspace(re_min, re_max, n_points_Re).astype(int)

#     print("Alfas:", alfas)
#     print("Re's:", Re)
    
    print('Prepare %s...' % fname)

    # set up foil data arrays
    foil_array = np.zeros((n_foil_params, n_points_Re, n_points_alfa))
    pre_foil_array = np.zeros((n_foil_params, n_points_Re, n_points_alfa))

    # load foil coords from file
    try:
        x, y = read_airfoil_dat_file(Path(foils_dat_path, fname))
    except:
        raise Exception("W: Foil %s failed to read from file, skipped." % (fname))

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
    for num in range(16):

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
        for alfa in range(32):
            foil_array[0, :, alfa] = fill_gaps_in_xfoil_curve(foil_array[0, :, alfa])
            foil_array[1, :, alfa] = fill_gaps_in_xfoil_curve(foil_array[1, :, alfa])
            foil_array[2, :, alfa] = fill_gaps_in_xfoil_curve(foil_array[2, :, alfa])
            foil_array[3, :, alfa] = fill_gaps_in_xfoil_curve(foil_array[3, :, alfa])            

    # save or discard data
    assert np.sum(np.isnan(foil_array))==0, "Что-то пошло не так"
    
    print('%s data prepared for prediction.' % fname)
    
    foil_array = foil_array[:n_layers, :, :]
    
    return foil_array, alfas, Re




def find_first_and_last_1_position(array):
    
    if np.sum(array)==0: return -1
    
    for first in range(len(array)):
        if array[first]!=0: break
        
    for last in range(len(array)-1,0,-1):
        if array[last]!=0: break
            
    return first, last    