# Installation Instructions for MLz_for_JWST

## Quick Start

### 1. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 2. Install External Dependencies

#### GPz++ (Required for GPz photo-z method)

1. Clone and compile the gpz++ repository:
```bash
git clone https://github.com/cschreib/gpzpp.git
cd gpzpp
make
```

2. Set the `GPZROOT` environment variable:
```bash
export GPZROOT=/path/to/gpzpp/
# Add to your ~/.zshrc or ~/.bashrc for persistence:
echo 'export GPZROOT=/path/to/gpzpp/' >> ~/.zshrc
```

3. Install the Python wrapper:
```bash
pip install gpz_pype
```

#### EAZY Photo-z (Required for template fitting method)

1. Install eazy-py:
```bash
pip install eazy-py
```

2. Download EAZY templates and filters (if not already present):
```bash
# The templates/ and FILTER.RES.latest should be symlinked to your working directory
# These are already included in the repository at the top level
```

#### Dust Maps (Required for galactic extinction correction)

1. Install dustmaps:
```bash
pip install dustmaps
```

2. Download the SFD dust map data:
```python
import dustmaps.sfd
dustmaps.sfd.fetch()
```

This downloads ~100MB of dust map data to `~/.dustmaps/`.

## Environment Setup

### Recommended: Use Conda Environment

```bash
# Create new conda environment
conda create -n mlz_jwst python=3.9
conda activate mlz_jwst

# Install conda-available packages first
conda install numpy scipy matplotlib astropy scikit-learn seaborn jupyter

# Install pip-only packages
pip install dustmaps gpz_pype eazy-py pyyaml tqdm urllib3
```

## Verify Installation

Run this in Python to verify all dependencies are available:

```python
import numpy as np
import astropy
import matplotlib
import sklearn
import scipy
import seaborn
import dustmaps
import gpz_pype
import eazy.photoz
from tqdm import tqdm

print("All dependencies successfully imported!")
```

## Data Requirements

The notebook requires spectroscopic and photometric catalogs:

1. **Spectroscopic catalogs** (in `data/`):
   - CANDELS HST catalogs (auto-downloaded from MAST if missing)
   - JWST/NIRSpec DJA v4.4: `dja_jwst_specz_v4.4.csv`
   - FRESCO emitters: `FRESCO_HA_emitters_release_v1.txt`, `FRESCO_O3_emitters_release_v1.txt`
   - ALT DR1: `ALT_DR1_public.fits`

2. **Photometric catalogs** (in `data/dja_cats/`):
   - DJA JWST photometry catalogs for 11 fields
   - Note: These are large files and may be gitignored

## Common Issues

### Issue: `ModuleNotFoundError: No module named 'gpz_pype'`
**Solution**: Install gpz_pype and ensure gpz++ is compiled and GPZROOT is set.

### Issue: `FileNotFoundError: FILTER.RES.latest not found`
**Solution**: Ensure `FILTER.RES.latest` and `templates/` are symlinked or present in the working directory.

### Issue: `dustmaps.sfd` fetch errors
**Solution**: Manually download dust maps or check internet connection:
```python
import dustmaps.sfd
dustmaps.sfd.fetch()
```

### Issue: EAZY import errors
**Solution**: Install eazy-py from GitHub if PyPI version is outdated:
```bash
pip install git+https://github.com/gbrammer/eazy-py.git
```

## Running the Notebook

```bash
jupyter notebook PaperCode.ipynb
```

Run cells sequentially from top to bottom. The notebook has dependencies between cells.

## Additional Resources

- GPz documentation: https://dunkenj.github.io/gpz_pype/
- EAZY documentation: https://github.com/gbrammer/eazy-py
- Dust maps: https://dustmaps.readthedocs.io/
