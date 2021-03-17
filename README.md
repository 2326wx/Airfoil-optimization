# Inverse design of airfoil with DL methods

## 1. Task description and problem setting.

**Task:** to create airfoil with required aerodynamical characteristics.

**Problem:** there is good XFoil tool for analysis of existing airfoil, but there is no tool, which can *create* arbitrary airfoil by known aerodynamical characteristics.

Foil looks like:
<img src = "https://github.com/2326wz/Airfoil-optimization/blob/master/images/mh32.png">

Characteristics looks like:
<img src = "https://github.com/2326wz/Airfoil-optimization/blob/master/images/mh32_polars.png">


**So we can get characteristics from foil picture but can not get new foil from required characteristics.**



## 2. Solution approach

Let's do like we usually do in DL: take some "black box", put known data **"X"** to the input, put desired data **"y"** to the output and run a training.

What do we want from this "black box?" New airfoil picture. So our **"y"** data will be foils images:
<img src = "https://github.com/2326wz/Airfoil-optimization/blob/master/images/foils.png">

Now we can take key airfoil characteristics and use as **"X"** data. What are requirements for this data?
- it shall be unique for each foil;
- it shall be full enough to describe airfoil;
- there shall be quite large and multidimensional array of such data to allow "black box" successfully train.

As soon as foil characteristics are unique for each combination of Re and Alpha, we easily get 3D array of **"X"** data:
- dimension 0: [list of required Res]
- dimension 1: [list of required Alfas]
- dimension 2: [list of required characteristics]



Such characteristics are:
*dimension 0:*
- Cl: lift coefficient;
- Cd: drag coefficient;
- Cm: moment coefficient;
- Cp: pressure coefficient.







# A collapsible section with markdown
<details>
  <summary>Click to expand!</summary>
  
  ## Heading
  1. A numbered
  2. list
     * With some
     * Sub bullets
</details>