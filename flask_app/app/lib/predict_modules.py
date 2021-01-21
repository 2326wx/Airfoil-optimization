import numpy as np

from xfoil import XFoil
from xfoil.model import Airfoil

from app.lib.preprocess_modules import *
from app.config import *


def prepare_foil_to_predict(fname, n_layers=8):
    '''
    Generates foil array, lists of Alfa and Re from .dat file with specified *fname* and config data.
    Returns foil_array(n_layers, n_points_Re, n_points_alfa)
    
    n_layers: layers to include [Cy, Cx, Cm, Cp, d, S, Re, Alfa]
    '''
    
    # get list of alfas and alfa step
    alfa_step, alfas = get_alfa_step(alfa_min, alfa_max, n_points_alfa)

    # set list of Re's
    Re = np.linspace(re_min, re_max, n_points_Re).astype(int)
   
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

    # store in array
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
    assert np.sum(np.isnan(foil_array))==0, "Foil array is empty!"
    
    print('%s data prepared for prediction.' % fname)
    
    foil_array = foil_array[:n_layers, :, :]
    
    return foil_array, alfas, Re




def find_first_and_last_1_position(array):
    '''
    Finds non-zero pixels numbers in 1D array
    Output: tuple of (first, last) non-zero pixels.
    On empty input returns -1.
    '''
    
    assert isinstance(array, np.ndarray), 'Image is not Numpy array'
    assert len(array.shape)==1, 'Not an 1D array'
    
    if np.sum(array)==0: return -1
    
    for first in range(len(array)):
        if array[first]!=0: break
        
    for last in range(len(array)-1,0,-1):
        if array[last]!=0: break
            
    return first, last    





def get_foil_xy_from_picture(y):
    '''
    Create arrays of X and Y foil coordinates for .dat file.
    
    Action:
    - find first and last point;
    - trim y array;
    - create lists with x and y coords;
    - remove NaN points;
    - interpolate gaps;
    - smooth foil.
    
    X = [1...0...1] along foil shape, Y = [-1...1]
    
    Input: 2D binary array == BW foil picture.
    Output: (x, y) arrays.
    '''
    
    assert isinstance(y, np.ndarray), 'Image is not Numpy array'
    assert len(y.shape)==2, 'Image is not binary'

    # сначала определим координаты начала и конца оси профиля

    for x_nose in range(y.shape[1]):
        res = find_first_and_last_1_position(y[:, x_nose])
        if res!=-1:
            y_nose_top, y_nose_bot = res
            break

    for x_tail in range(y.shape[1]-1,0,-1):
        res = find_first_and_last_1_position(y[:, x_tail])
        if res!=-1:
            y_tail_top, y_tail_bot = res
            break

    if y_nose_top!=y_nose_bot: # ставим точку в середине носика, пригодится для красивой аппроксимации
        y_nose = int(np.average((y_nose_top, y_nose_bot)))
        y[y_nose,x_nose-1]=1
        x_nose-=1

    y=y[:, x_nose:x_tail+1]

    foil_x_top = (np.arange(y.shape[1])).tolist(); foil_x_top.reverse()
    foil_x_bot = foil_x_top.copy() 
    foil_y_top=[]
    foil_y_bot=[]

    # ищем верхнюю и нижнюю границы профиля
    for x in foil_x_top:
        if find_first_and_last_1_position(y[:, x])!=-1:
            y_t, y_b = find_first_and_last_1_position(y[:, x])
            foil_y_top.append(y_nose - y_t)
            foil_y_bot.append(y_nose - y_b)
        else:
            foil_y_top.append(None)
            foil_y_bot.append(None)

    # склеиваем верх и низ в списки foil_x_top и foil_н_top
    foil_x_bot = foil_x_bot[:-1]
    foil_x_bot.reverse()
    foil_y_bot = foil_y_bot[:-1]
    foil_y_bot.reverse()
    foil_x_top.extend(foil_x_bot)
    foil_y_top.extend(foil_y_bot)

    foil_x_top = np.array(foil_x_top)
    foil_y_top = np.array(foil_y_top)

    # итоговые списки с координатами
    foil_x = []
    foil_y = []

    # удаляем точки, где дыры в У
    for i in range(foil_x_top.shape[0]):
        if foil_y_top[i]!=None:
            foil_x.append(foil_x_top[i])
            foil_y.append(foil_y_top[i])

    # превращаем списки в массивы и нормируем к 1
    foil_x = np.array(foil_x)/(y.shape[1]-1)
    foil_y = np.array(foil_y)/(y.shape[1]-1)
    foil_x.shape, foil_y.shape
    
    # интерполируем в нужное число точек
    f_x, f_y = interpolate_airfoil(foil_x, foil_y, n_points_in_predicted_dat)
    
    # сглаживаем
    f_x, f_y = smooth_foil_xy(f_x, f_y)   
    
    return f_x, f_y