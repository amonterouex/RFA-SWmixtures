
This guide explains how to run the Python implementation of the RFA for multicomponent square-well / square-shoulder mixtures. The program supports
- Mixtures of one, two, three, or more species;

- Input from a text file (`input.dat`) or interactive input from the command line when no input file is supplied;

- Calculation of radial distribution functions, cavity functions and Fourier-space functions for selected particle pairs or all pairs.


# :ferris_wheel: 1. Dependencies
The source code is provided as a python script and has the following prerequisites
- python (tested in python 3.10.11).
	- numpy (tested in 1.23.2)
	- scipy (tested in 1.9.1)
- The two python files needed to run the calculation are *main.py* and *rfa_sw_mixture.py*. Keep them in the same directory.

i
# :airplane: 2. Basic usage

## :inbox_tray: 2.1. Input data

There are two ways to provide the input data to run the code:

### a) Read the parameters from an input file

Use `-i` or `--input` to provide the input filename. Given an input filename like the template one *input_ternary_example.dat* or *input_binary_example.dat*, you can run the code and provide the input file as

  ```python  main.py -i input.dat```

### b) Enter the parameters interactively 
If no input file is given, the program will ask for the parameters interactively. The program will then ask for the mixture parameters one by one in the terminal.

```python  main.py```

  
## :outbox_tray: 2.2. Output data

- Use `-o` or `--output` to choose the base name of the output files.  For example: ```python  main.py  input.dat  -o  case1.dat```  creates files

	- case1_solver.dat

	- case1_g.dat

	- case1_y.dat

	- case1_fourier.dat

- The extension supplied after `-o` is treated as a base-name extension. For example, both ```python  main.py  input.dat  -o  case1.dat``` and ```python  main.py  input.dat  -o  case1``` produce files whose base name is `case1`.

- If `-o` is omitted, the default base name is `output`.
 
- An output directory can also be included: ```python  main.py  input.dat  -o  results/case1.dat```. This directory is created automatically if necessary.

# :clipboard: 3. Input information

## :pencil: 3.1. Physical parameters
  
The main input quantities are:
- `mole_fractions`: Mole fractions `x_i`. They must be non-negative and sum to 1.

- `diameters`: Hard-core particle diameters `sigma_i`. All must be positive.

- `epsilon`: Symmetric `N x N` interaction-energy matrix `epsilon_ij`.

- `delta`: Common shell width `Delta`. It must be non-negative.

- `rho`: Number density. It must be non-negative.

- `beta`: Inverse temperature.

 The size of the mixture is inferred from the number of entries in `mole_fractions`. Therefore, for an `N`-component mixture: 

-  `mole_fractions` must contain `N` values;

-  `diameters` must contain `N` values;

-  `epsilon` must contain `N` rows and `N` columns.

  The optional `n_species` line, if present, must agree with `mole_fractions`.
  
## :mag_right: 3.2. Particle pairs

-  You can calculate all unique pairs :
 ``` pairs = all ```

- Or you can instead request only selected pairs. For example:
```pairs = 1-1 1-2 2-2```
  

Pair indices in the input and output files are 1-based, matching the usual mathematical notation. Note that the program treats `2-1` and `1-2` as the same unique pair and stores it as `1-2`.



## :straight_ruler: 3.3. Real-space grid

  The real-space output is controlled by:
```
rmax = 9.0
dr = 0.1
```
The program uses one common `r` grid for all selected particle pairs. The grid starts at the smallest contact distance `sigma_ij` among the selected pairs and extends approximately to `rmax` with spacing `dr`.

## :triangular_ruler: 3.4. Fourier-space grid

The Fourier-space output is controlled by:
```
qmin = 0.01
qmax = 50.0
dq = 0.1
```

## :hourglass: 3.5. Inverse Laplace method

Two values are accepted:
- ```inversion = euler``` is the faster option corresponding to the fast Euler inversion settings
or
- ```inversion = euler_plus``` uses the more accurate, more computationally expensive Euler inversion settings.

For routine calculations, `euler` is generally the natural starting point. `euler_plus` can be used when a more demanding inverse-Laplace calculation is desired.




# :chart_with_upwards_trend: 4. Output information

  Suppose the calculation is run as:
```bash
python  main.py  input.dat  -o  mixture.dat
```
and that the program creates
```text
mixture_solver.dat
mixture_g.dat
mixture_y.dat
mixture_fourier.dat
```
Each of the previous files contain the following information:

### a) `*_solver.dat`

  
This file contains the input/state information and the RFA solver results. It includes the physical information provided, solver information, and the RFA coefficientes of every ordered pair

###  b)`*_g.dat`

This file contains the radial distribution functions. The **first column is `r`**, and every following column is one particle pair.

For a ternary mixture with `pairs = all`, its header is:

```text
# Radial distribution functions g_ij(r)
# Species indices are 1-based.
# Columns: r g_1_1 g_1_2 g_1_3 g_2_2 g_2_3 g_3_3
```

### c) `*_y.dat`

 This file contains the cavity functions. The **first column is `r`**, and every following column is one particle pair.

For a ternary mixture with `pairs = all`:

```text
# Cavity functions y_ij(r)
# Species indices are 1-based.
# Columns: r y_1_1 y_1_2 y_1_3 y_2_2 y_2_3 y_3_3
```
Note that the `g` and `y` files use the same radial grid, so corresponding rows refer to the same value of `r`.


###  d) `*_fourier.dat`

This file contains the Fourier transforms `htilde_ij(q)`. The **first column is `q`**, and every following column is one particle pair.

For a ternary mixture with `pairs = all`:
```text
# Fourier transforms htilde_ij(q)
# Species indices are 1-based.
# Columns: q htilde_1_1 htilde_1_2 htilde_1_3 htilde_2_2 htilde_2_3 htilde_3_3
```