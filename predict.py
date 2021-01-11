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

from lib.utils import load_pkl, save_pkl
from lib.preprocess_modules import *
from lib.predict_modules import *
from config import *

from nets.nn import *

def predict(fname):
    
    model = light_param_net(3072)
    model.load_weights(str(Path('./weights', weights_file)))
    
    df = pd.read_excel(Path(os.getcwd(),fname))
    
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
    
    y = (model.predict(X[None,:]))[0, :, :, 0]
    
    y[y>=yellow_threshold]=1
    y[y<yellow_threshold]=0
    
    plt.figure(figsize=(16,16))
    plt.matshow(y,0)
    plt.savefig(Path(os.getcwd(),fname.replace(' desired.xls', ' predicted.png')))
    
    print("\nSomething predicted, yellow pixels:", np.sum(y))
    time.sleep(5)
    
    if np.sum(y)==0: return
    
    
    # ищем координаты

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

    if y_tail_top!=y_tail_bot: # а на хвостике не ставим 
        pass

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

    # склеиваем верх и низ
    foil_x_bot = foil_x_bot[:-1]
    foil_x_bot.reverse()
    foil_y_bot = foil_y_bot[:-1]
    foil_y_bot.reverse()
    foil_x_top.extend(foil_x_bot)
    foil_y_top.extend(foil_y_bot)

    foil_x_top = np.array(foil_x_top)
    foil_y_top = np.array(foil_y_top)

    foil_x = []
    foil_y = []

    # удаляем точки, где дыры в У
    for i in range(foil_x_top.shape[0]):
        if foil_y_top[i]!=None:
            foil_x.append(foil_x_top[i])
            foil_y.append(foil_y_top[i])

    # нормируем к 1
    foil_x = np.array(foil_x)/(y.shape[1]-1)
    foil_y = np.array(foil_y)/(y.shape[1]-1)
    foil_x.shape, foil_y.shape
    
    f_x, f_y = interpolate_airfoil(foil_x, foil_y, n_points_in_predicted_dat)
    plt.figure(figsize=(16,16))
    plt.plot(f_x, f_y, 'green'), plt.plot([0,0],[-0.5,0.5],'white');
    
    df = pd.DataFrame(np.array((f_x, f_y)).T, columns=[fname.replace('desired.xls', 'predicted at'), str(datetime.now())[:19]],dtype='float32')
    savename = fname.replace('desired.xls', 'predicted.dat')
    df.to_csv(Path(os.getcwd(),savename), index=None, sep=' ')
    
    # get alfa step
    alfa_step, _ = get_alfa_step(alfa_min, alfa_max, n_points_alfa)

    print("Alfas:", alfas)
    print("Re's:", Re)

    print('Generate new foil data array...')        
    foil_array = create_foil_array_from_dat_file(Path(os.getcwd(), savename), Re, alfas, alfa_min, alfa_max, alfa_step)        

    save_pkl(foil_array, Path(foils_pkl_path, savename.replace('.dat', '.pkl')))
    print('Foil data array saved as %s ' % Path(foils_pkl_path, savename.replace('.dat', '.pkl')))              

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
        print('Saving as', savename.replace('.dat', '.xlsx'))
        db.to_excel(savename.replace('.dat', '.xlsx'), sheet_name='predicted', index=False)
    except:
        print('\n\nFile',savename.replace('.dat', '.xlsx'),'was not saved, because it opened by user.')

    print('\n\nReady, file saved as:', savename.replace('.dat', '.xlsx'))

    time.sleep(2)


if __name__ == "__main__":
    fname=sys.argv[1]
    predict(fname)