from pathlib import Path
import numpy as np



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