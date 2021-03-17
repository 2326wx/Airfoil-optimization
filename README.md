# Inverse design of airfoil with DL methods

## 1. Task description and problem setting

**Task:** to create airfoil with required aerodynamical parameters.

**Problem:** there is good XFoil tool for analysis of existing airfoil, but there is no tool, which can *create* arbitrary airfoil by known aerodynamical parameters.

Foil looks like:
<img src = "https://github.com/2326wz/Airfoil-optimization/blob/master/images/mh32.png">

parameters looks like:
<img src = "https://github.com/2326wz/Airfoil-optimization/blob/master/images/mh32_polars.png">


**So we can get parameters from foil picture but can not get new foil from required parameters.**




## 2. Solution approach

Let's do like we usually do in DL: take some "black box", put known data **"X"** to the input, put desired data **"y"** to the output and run a training.

What do we want from this "black box?" New airfoil picture. So our **"y"** data will be foils images:
<img src = "https://github.com/2326wz/Airfoil-optimization/blob/master/images/foils.png">

Now we can take key airfoil parameters and use as **"X"** data. What are requirements for this data?
- it shall be unique for each foil;
- it shall be full enough to describe airfoil;
- there shall be quite large and multidimensional array of such data to allow "black box" be successfully trained.

Such parameters are:
- **Cl**: lift coefficient;
- **Cd**: drag coefficient;
- **Cm**: moment coefficient;
- **Cp**: pressure coefficient;
- **d**:  foil thickness at flap root;
- **S**:  foil max thickness.



As soon as foil parameters are unique for each combination of Re and Alpha (except **S** and **d**), we easily get 3D array of **"X"** data:
- axis 0: required parameters for particular Re and Alpha;
- axis 1: Re;
- axis 2: Alpha:

<details>
  <summary>What are Re and Alpha?!!</summary>
  
  ### Re:
  
  <img src = "https://github.com/2326wz/Airfoil-optimization/blob/master/images/re.png">
  
  In fact, it is proportional to airflow *speed*, because all other variables are *fixed* in our approach.
  
  
  
  
  ### Alpha:
  
  <img src = "https://github.com/2326wz/Airfoil-optimization/blob/master/images/AoA.jpg">
  
  ***

</details>

Resulting array looks like:
<img src = "https://github.com/2326wz/Airfoil-optimization/blob/master/images/3dc.jpg">

Layers 6 and 7 with Re and Alpha will not take part in predictions; they need for info transfer between modules and can be dropped or replaced with some additional foil parameters.

So, now we have **"X"** data as an array of (*n_data_layers, n_points_Re, n_points_alfa*) shape.




## 3. Implementation 

For check of this data approach let's take extracting part of U-net and train it on images of 512x512 size.

**ToDo:**
1. Change image size to 256x1024 and achieve the same or better results.
2. Use more complicated CNN architectures.
3. Implement ensemble of different CNNs.




## 4. Loss and Metrics

Now using simplest MSE both for metric and loss.

**ToDo:** 
1. Use Tversky loss function and add IoU metric.




## 5. Production use

***Backend*** implemented as a microservice, based on a Flask server.

***Frontend*** implemented as Excel VBA macros, interacting with the server via HTTP requests. Why Excel?
- potential tool users are not experienced PC users and Excel is the maximum of their knowledge;
- large tables with input data require complicated frontend;
- Excel gives flexibility in adding new modules like graphs and charts.




## 6. Predictions test

Let's try to predict airfoil with known geometry by its parameters.

This is input array for Cl:
<img src = "https://github.com/2326wz/Airfoil-optimization/blob/master/images/xls_mh32.png">


This is prediction result before rounding, smoothing and interpolation:
<img src = "https://github.com/2326wz/Airfoil-optimization/blob/master/images/rough_mh32.png">


And here is smoothed interpolated prediction result, compared with true foil:
<img src = "https://github.com/2326wz/Airfoil-optimization/blob/master/images/result1.png">





