from app import app
import os
from flask import request, send_file, abort
from app.config import *
from pathlib import Path
from app.dat_to_xls import get_foil_array
from app.predict import predict
from app.nets.nn import *

app.config['SEND_FILE_MAX_AGE_DEFAULT']=0
app.config["CACHE_TYPE"] = "null"

def cleanup(del_list):
     for f in os.listdir(files_folder):
        for d in del_list:
            if d in f:
                try:
                    os.remove(os.path.join(files_folder,f))
                except:
                    pass

# define model
model = light_param_net()
model.load_weights(str(Path('./app/weights', weights_file)))
    

@app.route('/')
def rt():   
    return 'Server ok'   

@app.route('/index')
def idx():
    return 'Server ok'

@app.route('/load_foil', methods=['POST']) # to check: curl -F "file=@e387 predicted.dat" http://localhost:5000/load_foil --output myfile.xlsx
def process_loaded_foil():
    
    if 'file' not in request.files: abort(405, description='POST: No file in request.')

    # cleanup old files
    cleanup(['.dat', 'loaded.xls'])

    file = request.files['file']

    if '.dat' in file.filename:
        try:
            file.save(os.path.join(files_folder, file.filename))            
            return get_foil_array(file.filename)            
        except Exception as ex:
            abort(500, description=str(ex))
    else:
        abort(405, description='POST: File type not allowed.')

    
@app.route('/predict_foil', methods=['POST']) # to check: curl -F "file=@e387 desired.dat" http://localhost:5000/predict_foil --output myfile.xlsx
def predict_foil():
     
    if 'file' not in request.files: abort(405, description='POST: No file in request.')

    # cleanup old files
    cleanup(['desired.xls', 'predicted.xls', '.dat', '.png'])

    file = request.files['file']

    if '.xls' in file.filename:                    
        file.save(os.path.join(files_folder, file.filename))
        try: 
            return predict(file.filename, model)     
        except Exception as ex:
            abort(500, description=str(ex))
    else:
        abort(405, description='POST: File type not allowed.')

@app.route('/get_predicted_files', methods=['GET', 'POST']) # to check: curl --header "Content-Type: application/json" -d "{\"filename\":\"mh32.xlsx\"}" http://localhost:5000/get_predicted_files --output "qwerty.xlsx"
def get_dat():

    if request.json:
        fname = request.json['filename']
        try:
            print("Sending", fname)            
            return send_file(Path(xls_folder, fname), as_attachment=True, cache_timeout=0)  
        except:
            abort(404, description=str('No file ' + fname))
    else:
        abort(405, description=str(request.method)+": JSON with ['filename'] required.")

    

    