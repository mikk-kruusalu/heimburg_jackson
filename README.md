# Heimburg-Jackson equation

This is some follow up work from my bachelor thesis.

Equation for mechanical wave propagation in the nerve axon with an added dispersive term and mindlin microstructure proposed in [1]

```math
\begin{align*}
u_{tt} &= [(c_{0}^{2} + pu + qu^{2})u_{x}]_x - h_{1}u_{xxxx} + h_{2}u_{xxtt} + a_1 \phi_x \\
\phi_{tt} &= c_1^2 \phi_{xx} - \eta^2 \phi - a_2 u_x
\end{align*}
```

## Usage

Create virtual environment with

```bash
python -m venv venv
```

activate it with `source venv/bin/activate` on Linux.

For solving the equation use `solve.py` that takes two mandatory arguments -- the configuration yaml file and the output file paths. The yaml files should be self-explanatory. The output files can be analysed with any hdf5 viewer, [for example](https://myhdf5.hdfgroup.org/).

Alternatively one can use the `describe.py` script to see the contents of the file. And `plot.py` script for plotting different graphs.

[1] K. Tamm, T. Peets, and J. Engelbrecht, ‘Mechanical waves in myelinated axons’, Biomech Model Mechanobiol, vol. 21, no. 4, pp. 1285–1297, Aug. 2022, doi: 10.1007/s10237-022-01591-4.
