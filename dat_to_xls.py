import pandas as pd
from config import *
from lib.utils import *
from lib.preprocess_modules import *
from pathlib import Path
import os
import sys
import time

def get_foil_array(fname):
    
    print('Search for:', fname)
    time.sleep(2)
    
    print('Search %s in current folder...' % fname)    
    if fname not in os.listdir():
        print('Search %s in %s...' % (fname, foils_dat_path))
        if fname in os.listdir(foils_dat_path):
            dat_path = foils_dat_path
        else:
            print('No dat file found')
            time.sleep(5)
            raise Exception('No dat file found')
    else:
        dat_path='./'        
    print('Found %s at %s.' % (fname, dat_path))
    
    print('Search ready foil pkl data for %s in current folder...' % fname)
    if fname.replace('.dat', '.pkl') not in os.listdir():
        print('Search ready foil pkl data for %s in %s...' % (fname, foils_pkl_path))
        if  fname.replace('.dat', '.pkl') in os.listdir(foils_pkl_path):
            pkl_path = foils_pkl_path
        else:
            pkl_path = None
    else:
        pkl_path='./'
    
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
    try:
        print('Saving as', fname.replace('.dat', '.xlsx'))
        db.to_excel(fname.replace('.dat', '.xlsx'), sheet_name='loaded', index=False)
    except:
        print('\n\nFile',fname.replace('.dat', '.xlsx'),'was not saved, because it opened by user.')
    
    print('\n\nReady, file saved as:', fname.replace('.dat', '.xlsx'))
    
    time.sleep(2)

    return True



if __name__ == "__main__":
    fname=sys.argv[1]
    get_foil_array(fname)
    