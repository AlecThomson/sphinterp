# sphinterp

Astronomical catalogue interpolation on the sphere with HEALPix

## Installation

```bash
pip install git+https://github.com/AlecThomson/sphinterp.git
```

## Usage

See example notebook in `example/example.ipynb`.

For nearest-neigbour interpolation:

```python
from sphinterp import nn_interp_hpx
help(nn_interp_hpx)
```

Or, for inverse-distance-squared weigted linear interpolation:

```python
from sphinterp import idw_interp_hpx
help(idw_interp_hpx)
```