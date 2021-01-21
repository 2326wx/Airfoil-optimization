import pandas as pd
from app.config import *
from app.lib.utils import *
from app.lib.preprocess_modules import *
from pathlib import Path
import flask
import os
import sys
import time

def get_foil_array(fname):
    
    ''' Seeks requested foil in database and in cwd. If foil pkl array exists, uses it, else calculates foil params.
    Saves result as xls file.
    
    fname: Foil .dat file name.
    
    '''

    output = {}
    
    print('Search for %s params...' % fname)
        
    dat_path = files_folder
        
    print('Search ready foil pkl data for %s in files folder...' % fname)
    if fname.replace('.dat', '.pkl') not in os.listdir(files_folder):
        print('Search ready foil pkl data for %s in %s...' % (fname, foils_pkl_path))
        if  fname.replace('.dat', '.pkl') in os.listdir(foils_pkl_path):
            pkl_path = foils_pkl_path
        else:
            pkl_path = None
    else:
        pkl_path=files_folder
    
    if pkl_path:
        print('Found %s at %s.' % (fname.replace('.dat', '.pkl'), pkl_path))
    
    # get list of alfas and alfa step
    alfa_step, alfas = get_alfa_step(alfa_min, alfa_max, n_points_alfa)

    # set list of Re's
    Re = np.linspace(re_min, re_max, n_points_Re).astype(int)

    print("Alfas:", alfas)
    print("Re's:", Re)
    
    if pkl_path:        
        print('Use found foil data array from', Path(pkl_path, fname.replace('.dat', '.pkl')))
        foil_array = load_pkl(Path(pkl_path, fname.replace('.dat', '.pkl')))                                      
    else:                     
        print('Generate new foil data array...')        
        foil_array = create_foil_array_from_dat_file(Path(dat_path, fname), Re, alfas, alfa_min, alfa_max, alfa_step)        
        if dat_path == './':
            save_pkl(foil_array, fname.replace('.dat', '.pkl'))
            print('Foil data array saved as %s in current folder' % fname.replace('.dat', '.pkl'))
        else:
            save_pkl(foil_array, Path(foils_pkl_path, fname.replace('.dat', '.pkl')))
            print('Foil data array saved as %s ' % Path(foils_pkl_path, fname.replace('.dat', '.pkl')))              
            
    assert isinstance(foil_array, dict), 'Foil array is not a dict.'
    
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

    fname = fname.replace('.dat', ' loaded.xlsx')

    try:
        print('Saving as', str(Path(files_folder, fname)))
        db.to_excel(Path(files_folder, fname), sheet_name='loaded', index=False)
        output['xlsx'] = fname
    except:
        output['server_error'] = print('\n\nFile', fname, 'was not saved, because it opened by user.')
    
    print('\n\nReady, file saved as:', fname)

    return output



if __name__ == "__main__":
    fname=sys.argv[1]
    get_foil_array(fname)
    