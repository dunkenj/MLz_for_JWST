import numpy as np
from astropy.table import Table
from astropy.coordinates import SkyCoord
import astropy.units as u

from dustmaps.sfd import SFDQuery
from scipy.interpolate import CubicSpline
from xml.dom import minidom
import urllib3

def calcStats(photoz, specz, p90=False):
    """ Calculate statistics for photo-z vs spec-z comparison

    Parameters
    ----------
    photoz : array-like
        Array of photometric redshifts.
    specz : array-like
        Array of spectroscopic redshifts.
    p90 : bool, optional
        If True, calculate the 90th percentile statistics. Default is False.
    
    Returns
    -------
    scatter : float
        Scatter of the redshift differences, either 90th percentile or NMAD.
    OLF1 : float
        Outlier fraction for fixed 0.15*(1+z) threshold.
    OLF2 : float
        Outlier fraction for threshold of 3 times the scatter.
    bias : float
        Bias of the photo-z sample.
    """

    cut = np.logical_and(photoz >= 0, specz >= 0.)
    cut = np.logical_and(cut, np.isfinite(photoz))
    cut = np.logical_and(cut, np.isfinite(specz))   
    cut = np.logical_and(cut, np.isfinite(photoz))

    photoz = photoz[cut]
    specz = specz[cut]

    dz = photoz - specz # Difference between photo-z and spec-z
    abs_dz = np.abs(dz)/(1+specz) # Normalized difference

    # 90th Percentile Stats 
    if p90:
        p90 = (abs_dz < np.percentile(abs_dz, 90.))
        scatter = np.sqrt(np.sum((dz[p90]/(1+specz[p90]))**2) / float(len(dz[p90])))
        bias = np.nanmedian(dz[p90]/(1+specz[p90]))

        scatter = sigma_90
    
    # NMAD Stats
    else:
        scatter = 1.48 * np.median( np.abs(dz - np.median(dz)) / (1+specz)) 
        bias = np.nanmedian(dz/(1+specz))

    ol1 = (abs_dz > 0.15)
    ol2 = (abs_dz > (3*scatter))
    OLF1 = np.sum( ol1 ) / float(len(dz))
    OLF2 = np.sum( ol2 ) / float(len(dz))
    
    ol1_s, ol2_s = np.invert(ol1), np.invert(ol2)

    return scatter, OLF1, OLF2, bias

# Function - log(1+z) scaling
def forward(x):
    """
    Forward transformation for log(1+z) scaling.
    
    Parameters
    ----------
    x : array-like
        Input values to be transformed.
    
    Returns
    -------
    transformed : array-like
        Transformed values using log(1+x).
    """
  
    transformed = np.log10(1 + x)
    
    return transformed

def inverse(x):
    """
    Inverse transformation for log(1+z) scaling.
    Parameters
    ----------
    x : array-like
        Input values to be transformed.

    Returns 
    -------
    transformed : array-like
        Transformed values using 10^x - 1.
    """
    transformed = 10**x - 1
    
    return transformed


def f99_extinction(wave):
    """
    Return Fitzpatrick 99 galactic extinction curve as a function of wavelength

    Parameters
    ----------
    wave : astropy Quantity
        Wavelength in microns.  
    Returns
    -------
    extinction : astropy Quantity
        Extinction in magnitudes at the given wavelength.

    Notes
    -----
    The function uses a cubic spline interpolation based on anchor points defined in the Fitzpatrick 1999 paper.

    """
    anchors_x = [0., 0.377, 0.820, 1.667, 1.828, 2.141, 2.433, 3.704, 3.846]
    anchors_y = [0., 0.265, 0.829, 2.688, 3.055, 3.806, 4.315, 6.265, 6.591]

    f99 = CubicSpline(anchors_x, anchors_y)
    output_x = (1 / wave.to(u.micron))
    extinction = f99(output_x)

    return extinction

def filter_xml(filt):
    """
    Retrieve filter properties for JWST NIRCam filters
    or HST ACS WFC filters from the SVO Filter Profile Service.

    Parameters
    ----------
    filt : str
        Filter name, e.g., 'F435W', 'F606W', 'F814W' for HST or 'F115W', 'F200W', etc. for JWST.    

    Returns
    -------
    fdict : dict
        Dictionary containing filter properties such as wavelength reference and unit.

    Notes
    -----
    The function fetches the filter profile from the SVO Filter Profile Service and parses the XML response to extract relevant parameters.

    """
    http = urllib3.PoolManager()

    if filt.lower() in ['f435w', 'f606w', 'f814w']:
        file = http.request("GET", f"http://svo2.cab.inta-csic.es/theory/fps/fps.php?ID=HST/ACS_WFC.{filt.upper()}")
    else:
        file = http.request("GET", f"http://svo2.cab.inta-csic.es/theory/fps/fps.php?ID=JWST/NIRCam.{filt.upper()}")

    data = minidom.parseString(file.data)
    file.close()
    
    params = data.getElementsByTagName('PARAM')
    fdict = dict([(d.getAttribute('name'), d.getAttribute('value')) for d in params])
    return fdict

def ebv_corr(catalogue, filters, ra_col='RA', dec_col='DEC'):
    """ 
    Correct fluxes in catalogue for Galactic Extinction
    using the SFD dust map.

    Parameters
    ----------
    catalogue : astropy Table
        Catalogue containing fluxes and coordinates.
    filters : list of str
        List of filter names for which to apply the correction.
    ra_col : str, optional
        Name of the column containing Right Ascension coordinates. Default is 'RA'.
    dec_col : str, optional
        Name of the column containing Declination coordinates. Default is 'DEC'.

    Returns
    -------
    catalogue : astropy Table
        Catalogue with corrected fluxes for the specified filters.

    Notes
    -----
    This function uses the SFD dust map to obtain the E(B-V) values for the coordinates in the catalogue.
    It then applies the Fitzpatrick 1999 extinction curve to correct the fluxes for each specified filter.
    The extinction correction is applied as a multiplicative factor to the fluxes and flux errors in the catalogue.
    The corrected fluxes are stored in new columns named `<filter>_flux` and `<filter>_fluxerr`.
    The function assumes that the catalogue has columns for fluxes and flux errors corresponding to the specified filters.

    Example
    -------
    >>> from astropy.table import Table
    >>> catalogue = Table({'RA': [10.684, 10.684], 'DEC': [41.269, 41.269],
    ...                    'F435W_flux': [1.0, 2.0], 'F435W_fluxerr': [0.1, 0.2]})
    >>> filters = ['F435W', 'F606W']
    >>> corrected_catalogue = ebv_corr(catalogue, filters)
    >>> print(corrected_catalogue)
    >>> # Output will show the catalogue with corrected fluxes for F435W and F606W filters.

    """
    sfd = SFDQuery()
    coord = SkyCoord(catalogue[ra_col], catalogue[dec_col], unit='deg')
    ebv = sfd(coord)
    
    for i, filt in enumerate(filters): 
        print(filt)
        filt_dict = filter_xml(filt)
        filt_lambda = filt_dict['WavelengthRef']*u.Unit(filt_dict['WavelengthUnit'])

        a_lambda = f99_extinction(filt_lambda)

        mw_corr = 10**(a_lambda*ebv/2.5)

        catalogue[f'{filt}_flux'] *= mw_corr
        catalogue[f'{filt}_fluxerr'] *= mw_corr

    return catalogue

def find_ci_cut(pz, zgrid):
    """
    Find the confidence interval cut for a given probability distribution function (PDF) of redshift.
    This function iteratively reduces the PDF until 80% of the total area under the curve is contained within the cut.
    Parameters
    ----------
    pz : array-like
        Probability distribution function of redshift.
    zgrid : array-like
        Grid of redshift values corresponding to the PDF.

    Returns
    -------
    pz_c : float
        The cut value for the PDF such that 80% of the total area under the curve is contained within this cut.

    """

    peak_p = np.max(pz)
    pz_c = np.max(pz)
    int_pz = 0.

    while int_pz < 0.8: 
        pz_c *= 0.99
        cut = (pz < pz_c)
        pz_i = np.copy(pz)
        pz_i[cut] = 0.

        int_pz = np.trapezoid(pz_i, zgrid) / np.trapezoid(pz, zgrid)

    return pz_c


def get_peak_z(pz, zgrid):
    """
    Find the peaks in the probability distribution function (PDF) of redshift and calculate their properties.

    This function identifies the regions where the PDF exceeds a certain confidence interval cut (80% of the total area)
    and calculates the peak redshift, lower and upper bounds, and area under the curve for each peak.

    Parameters
    ----------
    pz : array-like
        Probability distribution function of redshift.
    zgrid : array-like  
        Grid of redshift values corresponding to the PDF.
    Returns
    -------
    zpeaks : array-like
        Array of peak redshift values.
    z_low : array-like
        Array of lower bounds for each peak.
    z_high : array-like
        Array of upper bounds for each peak.
    peak_areas : array-like
        Array of areas under the curve for each peak. 
    

    """

    pz_c = find_ci_cut(pz, zgrid)
    p80 = (pz > pz_c)

    pz_to_int = np.copy(pz)
    pz_to_int[np.invert(p80)] *= 0.

    lbounds = np.argwhere(np.diff(p80.astype('float')) == 1.).squeeze()
    ubounds = np.argwhere(np.diff(p80.astype('float')) == -1.).squeeze()
    lbounds = np.array(lbounds, ndmin=1)
    ubounds = np.array(ubounds, ndmin=1)

    zpeaks = []
    z_low = []
    z_high = []
    peak_areas = []

    if len(lbounds) >= 1 and len(ubounds) >= 1:
        if lbounds[0] > ubounds[0]:
            lbounds = np.insert(lbounds, 0, 0)

        for ipx in range(len(lbounds)):
            lo = lbounds[ipx]
            try:
                up = ubounds[ipx]
            except:
                up = -1

            area = np.trapezoid(pz_to_int[lo:up], zgrid[lo:up])
            peak_areas.append(area)

            zmz = np.trapezoid(pz_to_int[lo:up]*zgrid[lo:up], zgrid[lo:up]) / area
            z_low.append(zgrid[lo])
            z_high.append(zgrid[up])
            zpeaks.append(zmz)

        peak_areas = np.array(peak_areas)
        zpeaks = np.array(zpeaks)
        z_low = np.array(z_low)
        z_high = np.array(z_high)

        order = np.argsort(peak_areas)[::-1]

        return zpeaks[order], z_low[order], z_high[order], peak_areas[order]

    elif len(lbounds) == 1 and len(ubounds) == 0:
        lo = lbounds[0]
        area = np.trapezoid(pz_to_int[lo:-1], zgrid[lo:-1])
        peak_areas.append(area)

        zmz = np.trapezoid(pz_to_int[lo:-1]*zgrid[lo:-1], zgrid[lo:-1]) / area
        zpeaks.append(zmz)
        z_low.append(zgrid[lo])
        z_high.append(zgrid[-1])

        return zpeaks, z_low, z_high, peak_areas


    elif len(lbounds) == 0 and len(ubounds) == 1:
        up = ubounds[0]
        area = np.trapezoid(pz_to_int[0:up], zgrid[0:up])
        peak_areas.append(area)

        zmz = np.trapezoid(pz_to_int[0:up]*zgrid[0:up], zgrid[0:up]) / area
        zpeaks.append(zmz)
        z_low.append(zgrid[0])
        z_high.append(zgrid[up])

        return zpeaks, z_low, z_high, peak_areas

    else:
        # No peaks found        

        return -99., -99., -99., -99.

def pz_to_catalog(pz, zgrid, catalog, verbose=True):
    """
    Convert a probability distribution function (PDF) of redshift into a catalog format with peak redshift and confidence intervals.
    This function processes the PDF of redshift, identifies the peaks, and calculates the median, upper, and lower bounds for each peak.

    Parameters
    ----------
    pz : array-like
        Probability distribution function of redshift.
    zgrid : array-like
        Grid of redshift values corresponding to the PDF.   
    catalog : astropy Table
        Catalogue containing the IDs and spectroscopic redshifts.
    verbose : bool, optional
        If True, print progress messages. Default is True.
    Returns 
    -------
    output : astropy Table
        Catalogue with additional columns for peak redshift, lower and upper bounds, and area under the curve for each peak.

    """

    output = Table()
    pri_peakz = np.zeros_like(catalog['z_spec'])
    pri_upper = np.zeros_like(catalog['z_spec'])
    pri_lower = np.zeros_like(catalog['z_spec'])
    pri_area = np.zeros_like(catalog['z_spec'])

    pri_peakz.name = 'z1_median'
    pri_upper.name = 'z1_max'
    pri_lower.name = 'z1_min'
    pri_area.name = 'z1_area'

    pri_peakz.format = '%.4f'
    pri_upper.format = '%.4f'
    pri_lower.format = '%.4f'
    pri_area.format = '%.3f'


    sec_peakz = np.zeros_like(catalog['z_spec'])
    sec_upper = np.zeros_like(catalog['z_spec'])
    sec_lower = np.zeros_like(catalog['z_spec'])
    sec_area = np.zeros_like(catalog['z_spec'])

    sec_peakz.name = 'z2_median'
    sec_upper.name = 'z2_max'
    sec_lower.name = 'z2_min'
    sec_area.name = 'z2_area'

    sec_peakz.format = '%.4f'
    sec_upper.format = '%.4f'
    sec_lower.format = '%.4f'
    sec_area.format = '%.3f'
    
    for i, pzi in enumerate(pz):
        peaks, l80s, u80s, areas = get_peak_z(pzi, zgrid)
        peaks = np.array(peaks, ndmin=1)
        l80s = np.array(l80s, ndmin=1)
        u80s = np.array(u80s, ndmin=1)
        areas = np.array(areas, ndmin=1)

        if np.isnan(peaks[0]):
            pri_peakz[i] = -99.
        else:
            pri_peakz[i] = peaks[0]

        pri_upper[i] = u80s[0]
        pri_lower[i] = l80s[0]
        pri_area[i] = areas[0]

        if len(peaks) > 1:
            sec_peakz[i] = peaks[1]
            sec_upper[i] = u80s[1]
            sec_lower[i] = l80s[1]
            sec_area[i] = areas[1]
        else:
            sec_peakz[i] = -99.
            sec_upper[i] = -99.
            sec_lower[i] = -99.
            sec_area[i] = -99.

            
    pz_peak = zgrid[np.argmax(pz, axis=1)]

    output.add_column(catalog['ID'])
    output.add_column(catalog['z_spec'])
    output['z_peak'] = pz_peak
    output['z_peak'].format = '%.4f'

    output.add_column(pri_peakz)
    output.add_column(pri_lower)
    output.add_column(pri_upper)
    output.add_column(pri_area)

    output.add_column(sec_peakz)
    output.add_column(sec_lower)
    output.add_column(sec_upper)
    output.add_column(sec_area)
    return output