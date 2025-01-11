# Heimburg-Jackson equation

This is some follow up work from my bachelor thesis.

Equation for mechanical wave propagation in the nerve axon with an added dispersive term

```math
u_{tt} = [(c_{0}^{2} + pu + qu^{2})u_{x}]_x - h_{1}u_{xxxx} + h_{2}u_{xxtt}
```

## Usage

Create virtual environment with

```bash
python -m venv venv
```

activate it with `source venv/bin/activate` on Linux.

For solving the equation use `solve.py` that takes two mandatory arguments -- the configuration yaml file and the output file paths. The yaml files should be self-explanatory. The output files can be analysed with any hdf5 viewer, [for example](https://myhdf5.hdfgroup.org/).
