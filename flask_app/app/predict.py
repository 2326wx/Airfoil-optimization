import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import time
import os
import sys

from datetime import datetime

from xfoil import XFoil
from xfoil.model import Airfoil

from app.lib.utils import load_pkl, save_pkl
from app.lib.preprocess_modules import *
from app.lib.predict_modules import *
from app.config import *

from app.nets.nn import *

def predict(fname, model):
    ''' Predicts foil.
    Inputs: xls sheet with foil params with *fname*.
    Outputs: .dat and /xls files.
    '''    
    # dict for output file names
    output={}

    # load foil data from table
    df = pd.read_excel(os.path.join(files_folder, fname))
    # return(os.path.join('./app', xls_folder, fname))
    
    # get foil params
    Re = df.Re.unique().tolist()
    alfas = df.columns[4:].tolist()
    S = df.iloc[0,2]
    d = df.iloc[0,3]
    
    print('What in "%s"?\n' % fname)
    print("Available Re's:", Re)
    print("Available alfas:", alfas)
    print("Required S:", S)
    print("Required d:", d)
    
    foil_array = np.zeros((n_foil_params, n_points_Re, n_points_alfa), dtype='float64')
    foil_array[0,:,:] = df.iloc[0:16,4:]
    foil_array[1,:,:] = df.iloc[16:32,4:]
    foil_array[2,:,:] = df.iloc[32:48,4:]
    foil_array[3,:,:] = df.iloc[48:64,4:]
    foil_array[4,:,:] = d
    foil_array[5,:,:] = S
    foil_array[6,:,:] = (np.array((Re))*np.ones((32,16))).T
    foil_array[7,:,:] = (np.array((alfas))*np.ones((16,32)))
    
    foil_array = foil_array[:6,...]
    
    X = foil_array.reshape(foil_array.shape[0]*foil_array.shape[1]*foil_array.shape[2])
    re_dict = dict(zip(Re, range(len(Re))))
    
    # predict
    y = (model.predict(X[None,:]))[0, :, :, 0]
    
    # round plot
    y[y>=yellow_threshold]=1
    y[y<yellow_threshold]=0
    
    plt.figure(figsize=(16,16))
    plt.matshow(y,0)
    plt.savefig(Path(os.getcwd(), files_folder, fname.replace(' desired.xls', ' predicted.png')))
    output['png'] = str(fname.replace(' desired.xls', ' predicted.png'))
    
    print("\nSomething predicted, yellow pixels:", np.sum(y))

    if np.sum(y)==0: return  
       
    # smooth foil, get its coordinates
    f_x, f_y = get_foil_xy_from_picture(y)

    # save foil as .dat file
    df = pd.DataFrame(np.array((f_x, f_y)).T, columns=[fname.replace('desired.xls', 'predicted at'), str(datetime.now())[:19]],dtype='float32')
    savename = fname.replace('desired.xls', 'predicted.dat')
    df.to_csv(os.path.join('./app', xls_folder, savename), index=None, sep=' ')
    output['dat'] = savename
    
    # now calculate foil with XFoil and save foil data as xls file
    
    # get alfa step
    alfa_step, _ = get_alfa_step(alfa_min, alfa_max, n_points_alfa)

    print("Alfas:", alfas)
    print("Re's:", Re)

    print('Generate new foil data array...')        
    foil_array = create_foil_array_from_dat_file(Path(files_folder, savename), Re, alfas, alfa_min, alfa_max, alfa_step) 
    
    assert isinstance(foil_array, dict), 'Foil array is not a dict.'

    save_pkl(foil_array, Path(foils_pkl_path, savename.replace('.dat', '.pkl')))
    print('Foil data array saved as %s ' % Path(foils_pkl_path, savename.replace('.dat', '.pkl')))            

    foil_array = foil_array['X']
    S = foil_array[5,0,0]
    d = foil_array[4,0,0]

    db = np.zeros((4*len(Re), 4+len(alfas)))
    idx=['Param', 'Re', 'S', 'd']
    for i in range(len(alfas)): idx.append(str(round(alfas[i],2)))
    db = pd.DataFrame(db, columns=idx)

    abs_idx=0

    params = dict(zip(range(4),['Cy', 'Cx', 'Cm', 'Cp']))

    for param in params.keys():
        layer = param
        for re_num, re in enumerate(Re):
            db.iloc[abs_idx, 0]=params[param]
            db.iloc[abs_idx, 1]=re
            db.iloc[abs_idx, 2]=S
            db.iloc[abs_idx, 3]=d
            db.iloc[abs_idx, 4:]=foil_array[layer, re_num, :]            
            abs_idx+=1
    try:
        print('Saving as', savename.replace('.dat', '.xlsx'))
        db.to_excel(os.path.join('./app', xls_folder, savename.replace('.dat', '.xlsx')), sheet_name='predicted', index=False)
    except:
        print('\n\nFile',savename.replace('.dat', '.xlsx'),'was not saved, because it opened by user.')

    print('\n\nReady, file saved as:', savename.replace('.dat', '.xlsx'))

    output['xlsx'] = str(savename.replace('.dat', '.xlsx'))

    return output


if __name__ == "__main__":
    fname=sys.argv[1]
    predict(fname)