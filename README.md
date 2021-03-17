# Inverse design of airfoil with DL methods

## 1. Task description and problem setting.

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
- there shall be quite large and multidimensional array of such data to allow "black box" successfully train.

Such parameters are:
- Cl: lift coefficient;
- Cd: drag coefficient;
- Cm: moment coefficient;
- Cp: pressure coefficient;
- S:  foil max thickness;
- d:  foil thickness at flap root.


As soon as foil parameters are unique for each combination of Re and Alpha (except S and d), we easily get 3D array of **"X"** data:
- axis 0: required parameters for particular Re and Alpha;
- axis 1: Re;
- axis 2: Alpha:

<details>
  <summary>What are Re and Alpha?!!</summary>
  
  ### Re:
  
  <img src = "https://github.com/2326wz/Airfoil-optimization/blob/master/images/re.png">
  
  In fact, is proportional to airflow *speed*.
  
  
  
  ### Alpha:
  
  <img src = "https://github.com/2326wz/Airfoil-optimization/blob/master/images/AoA.jpg">
  
  ***

</details>

Resulting array looks like:
<img src = "https://github.com/2326wz/Airfoil-optimization/blob/master/images/3dc.jpg">

Layers 6 and 7 with Re and Alpha will not take part in predictions; they need for info transfer between modules and can be dropped or replaced with some additional foil parameters.


n_data_layers, n_points_Re, n_points_alfa










# A collapsible section with markdown
