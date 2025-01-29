[//]: # ([![Documentation Status]&#40;https://readthedocs.org/projects/adopt-net0/badge/?version=latest&#41;]&#40;https://adopt-net0.readthedocs.io/en/latest/?badge=latest&#41;)

[//]: # (![Testing]&#40;https://github.com/UU-ER/AdOpT-NET0/actions/workflows/00deploy.yml/badge.svg?branch=develop&#41;)

[//]: # ([![codecov]&#40;https://codecov.io/gh/UU-ER/AdOpT-NET0/graph/badge.svg?token=RVR402OGG0&#41;]&#40;https://codecov.io/gh/UU-ER/AdOpT-NET0&#41;)

[//]: # ([![Code style: black]&#40;https://img.shields.io/badge/code%20style-black-000000.svg&#41;]&#40;https://github.com/psf/black&#41;)

[//]: # ([![PyPI version]&#40;https://badge.fury.io/py/adopt-net0.svg&#41;]&#40;https://pypi.org/project/adopt-net0/&#41;)

[//]: # ([![status]&#40;https://joss.theoj.org/papers/12578885161d419241e50c5e745b7a11/status.svg&#41;]&#40;https://joss.theoj.org/papers/12578885161d419241e50c5e745b7a11&#41;)

[//]: # ([![DOI]&#40;https://zenodo.org/badge/DOI/10.5281/zenodo.13384688.svg&#41;]&#40;https://doi.org/10.5281/zenodo.13384688&#41;)
[//]: # ()
[//]: # (# AdOpT-NET0 - Advanced Optimization Tool for Networks and Energy)

[//]: # ()
[//]: # (This is a python package to simulate and optimize multi energy systems. It can )

[//]: # (model conversion technologies and networks for any carrier and optimize the )

[//]: # (design and operation of a multi energy system.)

[//]: # ()
[//]: # (## Installation)

[//]: # (You can use the standard utility for installing Python packages by executing the)

[//]: # (following in a shell:)

[//]: # ()
[//]: # (```pip install adopt_net0```)

[//]: # ()
[//]: # (Additionally, you need a [solver installed, that is supported by pyomo]&#40;https://pyomo.readthedocs.io/en/stable/solving_pyomo_models.html#supported-solvers&#41;)

[//]: # (&#40;we recommend gurobi, which has a free academic licence&#41;.)

[//]: # ()
[//]: # (Note for mac users: The export of the optimization results require a working)

[//]: # ([hdf5 library]&#40;https://www.hdfgroup.org/solutions/hdf5/&#41;. On windows this should be)

[//]: # (installed by default. On mac, you can install it with homebrew:)

[//]: # ()
[//]: # (```brew install hdf5```)

[//]: # ()
[//]: # (## Usage and documentation)

[//]: # (The documentation and minimal examples of how to use the package can be found )

[//]: # ([here]&#40;https://adopt-net0.readthedocs.io/en/latest/index.html&#41;. We also provide a )

[//]: # ([visualization tool]&#40;https://resultvisualization.streamlit.app/&#41; that is compatible )

[//]: # (with AdOpT-NET0.)

[//]: # ()
[//]: # (## Dependencies)

[//]: # (The package relies heavily on other python packages. Among others this package uses:)

[//]: # ()
[//]: # (- [pyomo]&#40;https://github.com/Pyomo/pyomo&#41; for compiling and constructing the model)

[//]: # (- [pvlib]&#40;https://github.com/pvlib/pvlib-python&#41; for converting climate data into )

[//]: # (  electricity output)

[//]: # (- [tsam]&#40;https://github.com/FZJ-IEK3-VSA/tsam&#41; for the aggregation of time series)

[//]: # ()
[//]: # (## Credits)

[//]: # (This tool was developed at Utrecht University.)
