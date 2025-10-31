# MLz_for_JWST: Machine Learning Photometric Redshifts for JWST

## Project Overview

This is a photometric redshift (photo-z) estimation pipeline for JWST deep field data, combining three complementary methods:
1. **GPz** (Gaussian Process regression) via `gpz_pype` Python wrapper
2. **EAZY** (template fitting) via `eazy-photoz` 
3. **NNz** (Nearest Neighbor with chi-squared weighting)

The pipeline produces consensus photo-z estimates with calibrated uncertainties for sparse JWST fields with limited spectroscopic training data.

## Critical Dependencies

### External Packages (must be installed separately)
- **gpz_pype**: Python wrapper for gpz++ C++ code ([docs](https://dunkenj.github.io/gpz_pype/))
  - Requires `gpz++` binary compiled and `GPZROOT` environment variable set
  - Example usage: `set_gpz_path(os.getenv('GPZROOT')+'bin/gpz++')`
- **eazy-photoz**: Template-based photo-z code (path referenced in `Users/duncan/Astro/code/eazy-py/eazy-photoz/`)
  - Requires symlinked `templates/` and `FILTER.RES.latest` filter file
  - Import: `import eazy.photoz`

### Python Stack
- Astronomy: `astropy`, `dustmaps` (SFD galactic extinction)
- ML: `gpz_pype`, `sklearn` (BallTree for NNz, train_test_split)
- Viz: `matplotlib`, `seaborn`, `tqdm`

## Key Architectural Components

### 1. Data Flow Pipeline (PaperCode.ipynb)

**Input Data Sources** (`data/` directory):
- **Spectroscopic catalogs**: CANDELS HST (5 fields), JWST/NIRSpec (DJA v4.4), JWST/WFSS (FRESCO H-alpha/OIII emitters), ALT DR1
- **Photometric catalogs**: DJA JWST catalogs from `dja_cats/` (11 fields: CEERS, GDS, GDN, Abell2744, PRIMER-COSMOS, etc.)
- Cross-matching uses `astropy.coordinates.SkyCoord.match_to_catalog_sky()` with 0.4" tolerance

**Processing Stages**:
1. **Spectroscopic sample assembly** (cells 2-6): Merge multiple spec-z sources, prioritize high-quality redshifts
2. **Photometric catalog cross-matching** (cells ~7-10): Match spec-z to photometry via sky coordinates
3. **Magnitude transformation** (cells ~11-13): Convert flux (μJy) → asinh magnitudes (luptitudes) using `flux_to_lupt()`
4. **Galactic extinction correction**: Apply via `ebv_corr()` using SFD dust maps and Fitzpatrick99 extinction curve
5. **Training/test split**: 70% train, 20% validation, 10% test (reproducible with `random_seed=1234`)

### 2. GPz Workflow (Primary Method)

**Configuration Files** (`default/` directory):
- `.param` files: GPz++ parameter files (e.g., `jwst_gpz_basic.param`)
- `_train.txt` / `_test.txt`: ASCII catalogs with luptitudes
- `_model.dat`: Saved GP model parameters

**Key Parameters** (from `jwst_gpz_basic.param`):
```
BANDS = ^lupt_(f115w|f150w|f200w|f277w|f356w|f444w)$  # 6 NIRCam filters (regex)
NUM_BF = 75                                            # GP basis functions (optimized via grid search)
COVARIANCE = gpvd                                      # Global diagonal covariance
WEIGHTING_SCHEME = uniform                             # Alternative: 'balanced', '1/(1+z)'
OUTPUT_MIN = 0, OUTPUT_MAX = 15                        # Redshift range
```

**Running GPz**:
```python
gpz = GPz(param_file='path/to.param', ncpu=8)
results, paths = gpz.run_training(
    catalog,                           # astropy.Table or file path
    outdir='default/',
    basename='jwst_gpz_basic',
    mag_prefix='lupt_',                # Column naming convention
    error_prefix='lupterr_',
    total_basis_functions=75,
    test_fraction=0.1,
    valid_fraction=0.2,
    output_max=15,
    random_seed=1234,
    bash_script=False                  # Run directly, not via bash
)
```

**Output columns**: `value` (z_phot), `uncertainty` (σ_z), plus all input columns

### 3. EAZY Template Fitting

**Template Files** (`templates/` directory):
- Multiple template sets: `eazy_v1.3.spectra.param`, `fsps_full/`, `sfhz/`, etc.
- Specify via `.param` files listing SED templates

**Key EAZY Conventions**:
- Zero-points: `pz.zp = [0.876, 0.871, 0.903, 1.0, 1.077, 1.148]` (from DJA for consistency)
- Outputs `chi2_fit` array: shape `(n_objects, n_zgrid)` for full P(z) distributions
- Convert χ² → P(z): `pzarr = np.exp(-0.5 * chi2_fit)`

### 4. NNz (Nearest Neighbor)

**Implementation** (cells ~30-32):
```python
tree = BallTree(train_features, leaf_size=15)
dist, ind = tree.query(test_features, k=300)  # Find 300 nearest neighbors

# Chi-squared weighted photo-z:
chisq = (((norm*model - data)**2) / (err**2)).sum(1)
weights = np.exp(-0.5 * chisq)
z_nnz = np.average(neighbor_redshifts, weights=weights)
```

**Key Detail**: Add systematic flux uncertainty: `data_err = np.sqrt(err**2 + (0.05*data)**2)`

### 5. Consensus Photo-z (Hierarchical Bayesian)

**P(z) Calibration**:
- Raw P(z) distributions are often over-confident
- Calibrate via exponentiation: `pz_calibrated = pz_raw^(1/beta)` where beta is optimized per method
- Example betas: GPz (0.66), EAZY (0.35), NNz (0.3)

**Hierarchical Bayesian Combination**:
```python
def HBpz(pzarr, zgrid, pzbad, beta=2., fbad_min=0.05, fbad_max=0.15):
    # Marginalize over fbad (catastrophic outlier fraction)
    pzb = (fbad * pzbad) + (pzarr * (1 - fbad))
    return np.trapezoid(pzarr_fbad, fbad_range, axis=2)
```

### 6. P(z) Analysis Utilities (gpz4jwst_utils.py)

**Core Functions**:
- `calcStats(photoz, specz)`: Returns σ_NMAD, OLF (outlier fractions), bias
  - NMAD: `1.48 * median(|dz - median(dz)| / (1+z))`
  - OLF₀.₁₅: fraction with `|dz|/(1+z) > 0.15`
  
- `find_ci_cut(pz, zgrid)`: Iteratively finds 80% confidence interval threshold
  
- `get_peak_z(pz, zgrid)`: Identifies peaks in P(z), returns sorted by area
  - Returns: `zpeaks, z_low, z_high, peak_areas` (for primary & secondary peaks)
  
- `pz_to_catalog(pz, zgrid, catalog)`: Convert P(z) arrays to catalog columns
  - Outputs: `z_peak`, `z1_median`, `z1_min`, `z1_max`, `z1_area` (primary peak)
  - Secondary peak columns: `z2_median`, etc.

- `ebv_corr(catalogue, filters)`: Galactic extinction correction
  - Fetches E(B-V) from SFD dust maps via `dustmaps.sfd.SFDQuery`
  - Applies F99 extinction curve via `f99_extinction()` cubic spline interpolation

**Transformation Functions**:
- `forward(x)`: `log10(1 + x)` for log-scaled plotting
- `inverse(x)`: `10^x - 1` for inverse transform
- Used with `Ax.set_yscale('function', functions=(forward, inverse))`

## Development Workflows

### Running the Full Pipeline

1. **Set up environment variables**:
   ```bash
   export GPZROOT=/path/to/gpz++/
   ```

2. **Execute notebook sequentially**: `PaperCode.ipynb` cells must run in order due to data dependencies

3. **Key checkpoint cells**:
   - Cell ~4: Spectroscopic sample assembly → `photom` table with matched spec-z
   - Cell ~13: GPz training → `simple_run` results table
   - Cell ~20: EAZY template fitting → `pz` object with `chi2_fit`
   - Cell ~32: NNz + consensus → `pzarr_hyb` final P(z)

### Adding New Filters

1. Update filter list: `filts = ['f115w', 'f150w', ..., 'new_filter']`
2. Ensure photometry columns exist: `{filt}_flux`, `{filt}_fluxerr`, `{filt}_rms`
3. Update GPz BANDS regex: `^lupt_(f115w|...|new_filter)$`
4. Verify `FILTER.RES.latest` includes new filter transmission curve

### Debugging GPz Issues

- Set `verbose=True` in `gpz.run_training()` to see gpz++ stdout
- Check intermediate files in `outdir`: `*_train.txt`, `*_model.dat`
- Common issue: NaN values → ensure `check_nans == 0` before training
- Basis function optimization: Plot σ_NMAD vs NUM_BF (cell ~14)

### Plotting Conventions

**Redshift Comparison Plots**:
```python
Ax.set_yscale('function', functions=(forward, inverse))  # Log(1+z) scaling
Ax.set_xlim([0, 11])  # z range
zrange = np.linspace(0, 12, 100)
Ax.plot(zrange, zrange ± 0.15*(1+zrange), ':')  # ±15% outlier threshold
```

**P(z) Calibration Curves**:
- Use `calc_ci_dist()` to compute cumulative confidence vs. HPD interval
- Diagonal line = perfect calibration
- Above diagonal = over-confident, below = under-confident

## Project-Specific Conventions

### Naming Patterns
- **Luptitudes**: `lupt_<filter>`, `lupterr_<filter>` (asinh magnitudes with softening)
- **Spectroscopic redshifts**: `z_spec` (always this column name)
- **Photo-z columns**: `z_phot`, `value` (GPz), `z_peak` (EAZY max likelihood)
- **Output basenames**: `jwst_gpz_<variant>` (e.g., `jwst_gpz_basic`, `jwst_gpz_basic_size`)

### Data Quality Flags
- **Good sources**: `check_nans == 0` (no NaN magnitudes)
- **Redshift cuts**: `0 < z_spec < 15`
- **Magnitude cuts**: `lupt_f444w < 27.5` (flux limit for statistics)
- **Spectroscopic quality**: JWST/NIRSpec `grade == 3` only

### Random Seeds
- Always use `random_seed=1234` for reproducibility in train/test splits and GP initialization
- sklearn splits also use `random_state=42` or `1234`

### Cross-matching Tolerance
- **0.4 arcsec** for spectroscopy-to-photometry matching
- Implemented via `SkyCoord.match_to_catalog_sky()` → check `d2d < 0.4*u.arcsec`

## Output Locations

- **Figures**: `figures/*.pdf` (publication-quality PDFs)
- **GPz models**: `default/*_model.dat` (reusable with `REUSE_MODEL = 1`)
- **Intermediate catalogs**: `default/*_train.txt`, `default/*_test.txt`
- **Final photo-z catalogs**: Returned as `astropy.Table` objects (save externally as needed)

## Common Pitfalls

1. **Missing gpz++ binary**: Ensure `GPZROOT` is set and `gpz++` is compiled
2. **Template/filter mismatch**: EAZY needs `templates/` and `FILTER.RES.latest` symlinked to working directory
3. **NaN propagation**: Always check `np.isnan()` before ML training
4. **Overconfident P(z)**: Apply beta calibration (values < 1.0) before combining methods
5. **Column name mismatches**: GPz expects exact `{prefix}{filter}` naming (case-sensitive)
