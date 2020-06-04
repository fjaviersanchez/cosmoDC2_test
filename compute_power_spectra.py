import numpy as np
import astropy.table
import matplotlib.pyplot as plt
import GCRCatalogs
import pymaster as nmt
import treecorr as tc
import pyccl as ccl
import healpy as hp
from optparse import OptionParser

parser = OptionParser()
parser.add_option('--input-catalog', dest="catalog_name", type=str, help='GCR catalog name', default='cosmoDC2_v1.1.4_image')
parser.add_option('--nbins', dest="nbins", type=int, help='Number of bins for treecorrs cf', default=50)
parser.add_option('--mag-max', dest='max_mag', type=float, help='Maximum magnitude to query', default=26)
parser.add_option('--band', dest='band', type=str, help='Filter band', default='r')
parser.add_option('--nside', dest='nside', type=int, help='Nside for maps', default=1024)
parser.add_option('--nzbins', dest='nzbins', type=int, default=4, help='Number of redshift bins')
parser.add_option('--debug', dest='debug', default=False, action='store_true', help='Print debugging information')
args, _ = parser.parse_args()
# Open the catalog
gc = GCRCatalogs.load_catalog(args.catalog_name)
# Get the data
data = gc.get_quantities(['ra', 'dec', 'redshift', f'mag_{args.band}_lsst'], filters=[f'mag_{args.band}_lsst < {args.max_mag}'])
print('Catalog loaded with', len(data['ra']), 'objects')
photoz = data['redshift'] # For now using true redshift
# Create tomographic bins
redshift_cuts = [(photoz > 0.2*(i+1)) & (photoz < 0.2*(i+2)) for i in range(0, args.nzbins)]
# Generate cosmology object
cosmo = ccl.Cosmology(Omega_c=0.265-0.0448, Omega_b=0.0448, n_s=0.963, sigma8=0.8, h=0.71, Neff=3.04, matter_power_spectrum='emu')
# Create TreeCorr catalogs
cat = [tc.Catalog(ra=data['ra'][redshift_cuts[i]], dec=data['dec'][redshift_cuts[i]], ra_units='deg', dec_units='deg') for i in range(0, args.nzbins)]
# Generate uniform randoms
ra_rnd = np.min(data['ra'])+(np.max(data['ra'])-np.min(data['ra']))*np.random.random(size=len(data['ra']))
min_cth = np.min(np.sin(np.radians(data['dec'])))
max_cth = np.max(np.sin(np.radians(data['dec'])))
rnd_cth = min_cth + (max_cth-min_cth)*np.random.random(size=len(data['ra']))
dec_rnd = np.degrees(np.arcsin(rnd_cth))
nside_high_res = 4096
# We generate a high resolution map with all the objects
hpmap = np.bincount(hp.ang2pix(nside_high_res, data['ra'], data['dec'], lonlat=True), minlength=hp.nside2npix(nside_high_res))

if args.debug:
    hp.mollview(hpmap, title='All galaxies')
    plt.show()
pixnums_rnd = hp.ang2pix(nside_high_res, ra_rnd, dec_rnd, lonlat=True)
hpxmask = (hpmap>0).astype(float)
mask = np.in1d(pixnums_rnd, np.where(hpmap>0)[0])
if args.debug:
    hp.mollview(hpxmask, title='Mask')
    plt.show()
ra_rnd = ra_rnd[mask] # Mask the randoms to just use those random particles in the footprint
dec_rnd = dec_rnd[mask]
cat_rnd = tc.Catalog(ra=ra_rnd, dec=dec_rnd, ra_units='deg', dec_units='deg') # TreeCorr random catalog
rr = tc.NNCorrelation(min_sep=0.01, max_sep=10, nbins=args.nbins, metric='Arc', sep_units='deg', bin_slop=0.1)
rr.process(cat_rnd)
if args.debug:
    plt.figure()
    plt.scatter(cat_rnd.ra[::1000], cat_rnd.dec[::1000], s=0.1)
    plt.scatter(cat[0].ra[::1000], cat[0].dec[::1000], s=0.1)
    plt.show()
theta = np.exp(rr.meanlogr)
# Compute theoretical predictions
cls_th = []
w_th = []
ell_max = np.max([10000, 3*args.nside])
ells = np.arange(1, ell_max)

for i in range(0, args.nzbins):
    dndz, be = np.histogram(data['redshift'][redshift_cuts[i]], bins=100, range=(0,3))
    #plt.plot(0.5*(be[1:]+be[:-1]), dndz*1.0)
    tracer = ccl.NumberCountsTracer(cosmo, True, dndz=(0.5*be[:1]+0.5*be[:-1], dndz*1.0), bias=(0.5*be[:1]+0.5*be[:-1], np.ones(100)))
    cls_th.append(ccl.angular_cl(cosmo, tracer, tracer, ells))
    w_th.append(ccl.correlation(cosmo, ells, cls_th[i], theta))
    if args.debug and i==0:
        plt.figure()
        plt.loglog(theta, w_th[i])
        plt.xlabel(r'$\theta$ [deg]')
        plt.ylabel(r'$w(\theta)$')
        plt.show()

w_out = {}
print('Computing ACF')
for i in range(0,args.nzbins):
    dd = tc.NNCorrelation(min_sep=0.01, max_sep=10, nbins=50, metric='Arc', sep_units='deg', bin_slop=0.1)
    dr = tc.NNCorrelation(min_sep=0.01, max_sep=10, nbins=50, metric='Arc', sep_units='deg', bin_slop=0.1)
    dd.process(cat[i])
    dr.process(cat[i], cat_rnd)
    xi, xivar = dd.calculateXi(rr,dr)
    w_out[f'w_{i}'] = xi
    w_out[f'w_err_p_{i}'] = xivar
    w_out[f'w_th_{i}'] = w_th[i]
    if args.debug and i==0:
        plt.figure()
        plt.loglog(theta, xi)
        plt.loglog(theta, w_th[0])
        plt.show()

w_out['theta'] = theta
astropy.table.Table(w_out).write(f'acf_{args.catalog_name}.fits.gz', overwrite=True)

# Repeat in Fourier space

# Set up NaMaster
wsp = nmt.NmtWorkspace()
if args.nside != hp.get_nside(hpxmask):
    hpxmask = hp.ud_grade(hpxmask, nside_out=args.nside)
# Create healpix maps
hpmap = [np.bincount(hp.ang2pix(args.nside, data['ra'][redshift_cuts[i]], 
                     data['dec'][redshift_cuts[i]], lonlat=True), minlength=hp.nside2npix(args.nside)) for i in range(0, args.nzbins)]
delta_maps = np.zeros((args.nzbins, len(hpmap[0])))
hpxmask[hpxmask!=1]=0.
ndens = [np.sum(hpmap[i][hpxmask==1])/(np.count_nonzero(hpxmask)*hp.nside2pixarea(args.nside)) for i in range(0, args.nzbins)]
print('Number densities:', ndens)
for i in range(0, args.nzbins):
    delta_maps[i][hpxmask==1] = hpmap[i][hpxmask==1]/np.mean(hpmap[i][hpxmask==1]) - 1.
# Set up bins
bins = nmt.NmtBin(nside=args.nside, nlb=int(len(hpxmask)*1.0/np.count_nonzero(hpxmask)))
print('Computing power spectra')
cls_out = {}
for i in range(0, args.nzbins):
    f = nmt.NmtField(hpxmask, [delta_maps[i]])
    if i==0:
        wsp.compute_coupling_matrix(f, f, bins)
    cls_out[f'Cl_{i}'] = wsp.decouple_cell(nmt.compute_coupled_cell(f, f))[0]
    cls_out[f'Nl_{i}'] = wsp.decouple_cell(wsp.couple_cell([1./ndens[i]*np.ones(3*args.nside)]))[0]
    cls_out[f'Cl_th_{i}'] = wsp.decouple_cell(wsp.couple_cell([cls_th[i][:3*args.nside]]))[0]
    if i==0:
        cls_out['ells'] = bins.get_effective_ells()
    if args.debug and i==0:
        plt.figure()
        plt.loglog(cls_out['ells'], cls_out[f'Cl_{i}'])
        plt.loglog(cls_out['ells'], cls_out[f'Cl_th_{i}'])
        plt.show()

astropy.table.Table(cls_out).write(f'cls_{args.catalog_name}.fits.gz', overwrite=True)
