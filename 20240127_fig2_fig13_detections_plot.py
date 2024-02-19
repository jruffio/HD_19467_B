import numpy as np
import matplotlib.pyplot as plt
from  scipy.interpolate import interp1d
import os
import astropy.io.fits as fits
import h5py
from scipy.interpolate import RegularGridInterpolator
import astropy.units as u
from astropy import constants as const
from scipy.signal import correlate2d

import matplotlib.patheffects as PathEffects
from copy import copy
import matplotlib
matplotlib.use('Qt5Agg')

import matplotlib.gridspec as gridspec # GRIDSPEC !
from matplotlib.colorbar import Colorbar

def fill_gap(HD19467B_wvs,HD19467B_spec,planet_f,fit_range= [3.9,4.3]):


    HD19467B_wvs = HD19467B_wvs[np.where(np.isfinite(HD19467B_spec))]
    HD19467B_spec = HD19467B_spec[np.where(np.isfinite(HD19467B_spec))]
    HD19467B_dwvs = HD19467B_wvs[1::]-HD19467B_wvs[0:np.size(HD19467B_wvs)-1]
    HD19467B_dwvs = np.insert(HD19467B_dwvs,0,HD19467B_dwvs[0])
    startid_2ndhalf = np.where(HD19467B_dwvs>0.05)[0][0]
    dw = np.nanmedian(HD19467B_dwvs)
    gap_wvs = np.arange(HD19467B_wvs[startid_2ndhalf-1]+dw,HD19467B_wvs[startid_2ndhalf],dw)
    HD19467B_gapfilled_wvs = np.concatenate([HD19467B_wvs[0:startid_2ndhalf],gap_wvs,HD19467B_wvs[startid_2ndhalf::]])

    rv=0
    where_wvs4fit = np.where((HD19467B_wvs>fit_range[0])*(HD19467B_wvs<fit_range[1]))
    HD19467B_wvs_4fit = HD19467B_wvs[where_wvs4fit]
    HD19467B_spec_4fit = HD19467B_spec[where_wvs4fit]
    comp_spec_4fit = planet_f(HD19467B_wvs_4fit * (1 - (rv) / const.c.to('km/s').value)) * (u.W / u.m ** 2 / u.um)
    comp_spec_4fit = comp_spec_4fit * (HD19467B_wvs_4fit * u.um) ** 2 / const.c  # from  Flambda to Fnu
    comp_spec_4fit = comp_spec_4fit.to(u.MJy).value

    comp_spec_scale = np.nansum(HD19467B_spec_4fit*comp_spec_4fit)/np.nansum(comp_spec_4fit**2)

    comp_spec_gap = planet_f(gap_wvs * (1 - (rv) / const.c.to('km/s').value)) * (u.W / u.m ** 2 / u.um)
    comp_spec_gap = comp_spec_gap * (gap_wvs * u.um) ** 2 / const.c  # from  Flambda to Fnu
    comp_spec_gap = comp_spec_scale * comp_spec_gap.to(u.MJy).value
    HD19467B_gapfilled_spec = np.concatenate([HD19467B_spec[0:startid_2ndhalf],comp_spec_gap,HD19467B_spec[startid_2ndhalf::]])

    return HD19467B_gapfilled_wvs,HD19467B_gapfilled_spec

if __name__ == "__main__":
    try:
        import mkl
        mkl.set_num_threads(1)
    except:
        pass

    ####################
    ## To be modified
    ####################
    # for plotting
    fontsize = 12
    # Directories to update
    os.environ["WEBBPSF_PATH"] = "/stow/jruffio/data/webbPSF/webbpsf-data"
    crds_dir="/stow/jruffio/data/JWST/crds_cache/"
    # External_dir should external files like the NIRCam filters
    external_dir = "/stow/jruffio/data/JWST/external/"
    # output dir for images
    out_png = "/stow/jruffio/data/JWST/nirspec/HD_19467/breads/figures"
    # out_png = out_dir.replace("xy","figures")
    if not os.path.exists(out_png):
        os.makedirs(out_png)
    # Path to a s3d cube to extract an empirical PSF profile
    A0_filename = "/stow/jruffio/data/JWST/nirspec/A0_TYC 4433-1800-1/MAST_2023-04-23T0044/JWST/jw01128-o009_t007_nirspec_g395h-f290lp/jw01128-o009_t007_nirspec_g395h-f290lp_s3d.fits"
    ####################
    # RA Dec offset of the companion HD19467B
    ra_offset = -1332.871/1000. # ra offset in as
    dec_offset = -875.528/1000. # dec offset in as
        # Offsets for HD 19467 B from https://www.whereistheplanet.com/
        # RA Offset = -1332.871 +/- 10.886 mas
        # Dec Offset = -875.528 +/- 12.360 mas
        # Separation = 1593.703 +/- 9.530 mas
        # PA = 236.712 +/- 0.483 deg
        # Reference: Brandt et al. 2021
    # Absolute fluxes for the host star to be used in calculated flux ratios with the companion.
    HD19467_flux_MJy = {'F250M': 3.51e-06,
                        'F300M': 2.63e-06,
                        'F335M': 2.1e-06,
                        'F360M': 1.82e-06,
                        'F410M': 1.49e-06,
                        'F430M': 1.36e-06,
                        'F460M': 1.12e-06,
                        'F444W': 1.3170187436215383e-06,
                        'F356W': 1.925490551915828e-06,
                        'Lp': 1.7151743669362699e-06,
                        'Mp': 1.0371238943582984e-06}
    HD19467B_NIRSpec_flux_MJy = {'F360M': 6.326274647551914e-11,
                                 'F410M': 1.867789588572561e-10,
                                 'F430M': 1.3072878700327478e-10,
                                 'F460M': 8.779697757999878e-11,
                                 'F444W': 1.4276482620145773e-10,
                                 'F356W': 6.505583140450902e-11,
                                 'Lp': 1.1513475637429364e-10,
                                 'Mp': 9.208862832903471e-11}
    HD19467B_NIRSpec_fluxerr_MJy = {'F360M': 7.647496388253355e-12,
                                    'F410M': 7.429195302021038e-12,
                                    'F430M': 7.569167060980653e-12,
                                    'F460M': 7.88277602649061e-12,
                                    'F444W': 6.83433250731064e-12,
                                    'F356W': 5.018375164030435e-12,
                                    'Lp': 6.741786986251594e-12,
                                    'Mp': 6.8708378623151476e-12}
    color_list = ["#ff9900","#006699", "#6600ff", "#006699", "#ff9900", "#6600ff"]

    if 1: # NIRCam contrast for comparison
        F444W_Carter = np.loadtxt("/stow/jruffio/data/JWST/nirspec/HD_19467/breads/figures/HIP65426_ERS_CONTRASTS/F444W_ADI+RDI.txt")
        F444W_Carter_seps = F444W_Carter[:, 0]
        F444W_Carter_cons = F444W_Carter[:, 1]*np.sqrt(1236/2100*10**((5.4-6.771)/2.5))
        # plt.plot(F444W_Carter_seps,F444W_Carter_cons)
        # plt.yscale("log")
        # plt.show()


    if 1: # Load a cube to get the PSF image and PSF profile
        mast_s3d = "/stow/jruffio/data/JWST/nirspec/HD_19467/breads/figures/jw01414-o004_t003_nirspec_g395h-f290lp_s3d.fits"
        hdulist_sc = fits.open(mast_s3d)
        print(hdulist_sc.info())
        cube = hdulist_sc["SCI"].data
        # data = fits.open("/stow/jruffio/data/JWST/nirspec/A0_TYC 4433-1800-1/MAST_2023-06-22T1638_ifu/JWST/jw01128-o009_t007_nirspec_g395h-f290lp/jw01128-o009_t007_nirspec_g395h-f290lp_x1d.fits")
        # spectrum_info = data[1].data
        # wavelength = spectrum_info['WAVELENGTH']
        # id_4p5um = np.nanargmin(np.abs(wavelength-4.5))
        # HD19467_median_im = cube[id_4p5um]
        HD19467_median_im = np.nanmedian(cube,axis=0)[:,::-1]
        # plt.imshow(HD19467_median_im[:,::-1]/1e-12)
        # plt.show()
        kc,lc = 32,25# np.unravel_index(np.nanargmax(HD19467_median_im),HD19467_median_im.shape)
        s3d_Dec_vec = (np.arange(HD19467_median_im.shape[0])-kc)*0.1 #- ra_corr
        s3d_RA_vec = (np.arange(HD19467_median_im.shape[1])-lc)*0.1 #- dec_corr
        s3d_dDec = s3d_Dec_vec[1]-s3d_Dec_vec[0]
        s3d_dRA = s3d_RA_vec[1]-s3d_RA_vec[0]


        A0_filename = "/stow/jruffio/data/JWST/nirspec/A0_TYC 4433-1800-1/MAST_2023-04-23T0044/JWST/jw01128-o009_t007_nirspec_g395h-f290lp/jw01128-o009_t007_nirspec_g395h-f290lp_s3d.fits"
        hdulist_sc = fits.open(A0_filename)
        cube = hdulist_sc["SCI"].data
        im_psfprofile = np.nanmedian(cube,axis=0)
        im_psfprofile /= np.nanmax(im_psfprofile)
        # kc,lc = np.unravel_index(np.nanargmax(im_psfprofile),im_psfprofile.shape)
        kc, lc = 38.5,34.5
        kvec = (np.arange(im_psfprofile.shape[0])-kc)*0.1
        lvec = (np.arange(im_psfprofile.shape[1])-lc)*0.1
        lgrid,kgrid = np.meshgrid(lvec,kvec)
        rgrid_psfprofile = np.sqrt(lgrid**2+kgrid**2)
        im_psfprofile[np.where(rgrid_psfprofile>1.5)] = np.nan
        # plt.imshow(im_psfprofile,origin="lower")
        # plt.show()




    F460M_to_F444W = HD19467B_NIRSpec_flux_MJy["F444W"]/HD19467B_NIRSpec_flux_MJy["F460M"]
    nrs2_flux_filename = os.path.join(out_png,"HD19467B_FM_F460M_nrs2_flux.fits")
    with fits.open(nrs2_flux_filename) as hdulist:
        nrs2_flux = hdulist[0].data*F460M_to_F444W
    nrs2_fluxerr_filename = os.path.join(out_png,"HD19467B_FM_F460M_nrs2_fluxerr.fits")
    with fits.open(nrs2_fluxerr_filename) as hdulist:
        nrs2_fluxerr = hdulist[0].data*F460M_to_F444W
    nrs2_RADec_filename = os.path.join(out_png,"HD19467B_FM_F460M_nrs2_RADec.fits")
    with fits.open(nrs2_RADec_filename) as hdulist:
        nrs2_RA_grid = hdulist[0].data
        nrs2_Dec_grid = hdulist[1].data
        nrs2_RA_vec =   nrs2_RA_grid[0,:]
        nrs2_Dec_vec =   nrs2_Dec_grid[:,0]
    dra = nrs2_RA_vec[1]-nrs2_RA_vec[0]
    ddec = nrs2_Dec_vec[1]-nrs2_Dec_vec[0]
    rs_grid = np.sqrt(nrs2_RA_grid**2+nrs2_Dec_grid**2)
    PAs_grid = np.arctan2(nrs2_RA_grid,nrs2_Dec_grid) % (2 * np.pi)
    nrs2_hist_filename = os.path.join(out_png,"HD19467B_FM_F460M_nrs2_hist.fits")
    with fits.open(nrs2_hist_filename) as hdulist:
        nrs2_bins = hdulist[0].data
        nrs2_hist = hdulist[1].data

    F360M_to_F444W = HD19467B_NIRSpec_flux_MJy["F444W"]/HD19467B_NIRSpec_flux_MJy["F360M"]
    nrs1_flux_filename = os.path.join(out_png,"HD19467B_FM_F360M_nrs1_flux.fits")
    with fits.open(nrs1_flux_filename) as hdulist:
        nrs1_flux = hdulist[0].data*F360M_to_F444W
    nrs1_fluxerr_filename = os.path.join(out_png,"HD19467B_FM_F360M_nrs1_fluxerr.fits")
    with fits.open(nrs1_fluxerr_filename) as hdulist:
        nrs1_fluxerr = hdulist[0].data*F360M_to_F444W
    nrs1_RADec_filename = os.path.join(out_png,"HD19467B_FM_F360M_nrs1_RADec.fits")
    with fits.open(nrs1_RADec_filename) as hdulist:
        nrs1_RA_grid = hdulist[0].data
        nrs1_Dec_grid = hdulist[1].data
        nrs1_RA_vec =   nrs1_RA_grid[0,:]
        nrs1_Dec_vec =   nrs1_Dec_grid[:,0]
    nrs1_hist_filename = os.path.join(out_png,"HD19467B_FM_F360M_nrs1_hist.fits")
    with fits.open(nrs1_hist_filename) as hdulist:
        nrs1_bins = hdulist[0].data
        nrs1_hist = hdulist[1].data

    deno = (1/nrs1_fluxerr**2 + 1/nrs2_fluxerr**2)
    combined_flux = (nrs1_flux/nrs1_fluxerr**2 + nrs2_flux/nrs2_fluxerr**2)/deno
    combined_flux_err = 1/np.sqrt(deno)
    combined_SNR = combined_flux/combined_flux_err
    contrast_5sig_combined  = 5*combined_flux_err/HD19467_flux_MJy["F444W"]
    contrast_5sig_combined[np.where(~np.isfinite(contrast_5sig_combined))] = np.nan


    mask= (-0.6+1./6. * np.pi+PAs_grid)% (2./6. * np.pi)
    mask = np.abs(mask-1./6. * np.pi)<(np.pi/24)


    contrast_5sig_1D_seps = np.arange(0,3,0.1)
    contrast_5sig_1D_med = np.zeros(contrast_5sig_1D_seps.shape)+np.nan
    contrast_5sig_1D_min = np.zeros(contrast_5sig_1D_seps.shape)+np.nan
    contrast_5sig_1D_inspike = np.zeros(contrast_5sig_1D_seps.shape)+np.nan
    contrast_5sig_1D_outspike = np.zeros(contrast_5sig_1D_seps.shape)+np.nan
    for k,sep in enumerate(contrast_5sig_1D_seps):
        contrast_5sig_1D_med[k] = np.nanmedian(contrast_5sig_combined[np.where((rs_grid>(sep-0.05))*(rs_grid<(sep+0.05)))])
        contrast_5sig_1D_min[k] = np.nanmin(contrast_5sig_combined[np.where((rs_grid>(sep-0.05))*(rs_grid<(sep+0.05)))])
        contrast_5sig_1D_inspike[k] = np.nanmedian(contrast_5sig_combined[np.where((~mask)*(rs_grid>(sep-0.05))*(rs_grid<(sep+0.05)))])
        contrast_5sig_1D_outspike[k] = np.nanmin(contrast_5sig_combined[np.where(mask*(rs_grid>(sep-0.05))*(rs_grid<(sep+0.05)))])
    fm_r2comp_grid = np.sqrt((nrs1_RA_grid-ra_offset)**2+(nrs1_Dec_grid-dec_offset)**2)


    rdi_nrs2_flux_filename = os.path.join(out_png,"HD19467B_RDI_F460M_nrs2_flux.fits")
    with fits.open(rdi_nrs2_flux_filename) as hdulist:
        rdi_nrs2_flux = hdulist[0].data*F460M_to_F444W
    rdi_nrs2_fluxerr_filename = os.path.join(out_png,"HD19467B_RDI_F460M_nrs2_fluxerr.fits")
    with fits.open(rdi_nrs2_fluxerr_filename) as hdulist:
        rdi_nrs2_fluxerr = hdulist[0].data*F460M_to_F444W
    rdi_nrs2_RADec_filename = os.path.join(out_png,"HD19467B_RDI_F460M_nrs2_RADec.fits")
    with fits.open(rdi_nrs2_RADec_filename) as hdulist:
        rdi_nrs2_RA_grid = hdulist[0].data
        rdi_nrs2_Dec_grid = hdulist[1].data
        rdi_nrs2_RA_vec =   rdi_nrs2_RA_grid[0,:]
        rdi_nrs2_Dec_vec =   rdi_nrs2_Dec_grid[:,0]
    rdi_dra = rdi_nrs2_RA_vec[1]-rdi_nrs2_RA_vec[0]
    rdi_ddec = rdi_nrs2_Dec_vec[1]-rdi_nrs2_Dec_vec[0]
    rdi_rs_grid = np.sqrt(rdi_nrs2_RA_grid**2+rdi_nrs2_Dec_grid**2)

    rdi_nrs1_flux_filename = os.path.join(out_png,"HD19467B_RDI_F360M_nrs1_flux.fits")
    with fits.open(rdi_nrs1_flux_filename) as hdulist:
        rdi_nrs1_flux = hdulist[0].data*F360M_to_F444W
    rdi_nrs1_fluxerr_filename = os.path.join(out_png,"HD19467B_RDI_F360M_nrs1_fluxerr.fits")
    with fits.open(rdi_nrs1_fluxerr_filename) as hdulist:
        rdi_nrs1_fluxerr = hdulist[0].data*F360M_to_F444W
    rdi_nrs1_RADec_filename = os.path.join(out_png,"HD19467B_RDI_F360M_nrs1_RADec.fits")
    with fits.open(rdi_nrs1_RADec_filename) as hdulist:
        rdi_nrs1_RA_grid = hdulist[0].data
        rdi_nrs1_Dec_grid = hdulist[1].data
        rdi_nrs1_RA_vec =   rdi_nrs1_RA_grid[0,:]
        rdi_nrs1_Dec_vec =   rdi_nrs1_Dec_grid[:,0]


    rdi_deno = (1/rdi_nrs1_fluxerr**2 + 1/rdi_nrs2_fluxerr**2)
    rdi_combined_flux = (rdi_nrs1_flux/rdi_nrs1_fluxerr**2 + rdi_nrs2_flux/rdi_nrs2_fluxerr**2)/rdi_deno
    kmax,lmax = np.unravel_index(np.nanargmax(rdi_combined_flux*(((rdi_nrs1_RA_grid-(-1.5))**2+(rdi_nrs1_Dec_grid-(-0.9))**2)<0.4)),rdi_combined_flux.shape)
    rdi_r2comp_grid = np.sqrt((rdi_nrs1_RA_grid-rdi_nrs1_RA_grid[kmax,lmax])**2+(rdi_nrs1_Dec_grid-rdi_nrs1_Dec_grid[kmax,lmax])**2)
    rdi_r2star_grid = np.sqrt((rdi_nrs1_RA_grid)**2+(rdi_nrs1_Dec_grid)**2)
    # rdi_combined_flux_err = 1/np.sqrt(rdi_deno)
    rdi_combined_flux = rdi_combined_flux- np.nanmedian(rdi_combined_flux[np.where((rdi_r2star_grid > 0.5))])
    rdi_combined_SNR = rdi_combined_flux / ((rdi_nrs1_fluxerr+rdi_nrs2_fluxerr)/2.)
    rdi_combined_SNR[np.where((rdi_r2star_grid < 0.5))] = np.nan
    rdi_combined_SNR = rdi_combined_SNR / np.nanstd(rdi_combined_SNR[np.where(rdi_r2comp_grid > 0.4)])
    rdi_combined_flux_err = rdi_combined_flux/rdi_combined_SNR
    rdi_combined_SNR = rdi_combined_flux/rdi_combined_flux_err
    rdi_contrast_5sig_combined  = 5*rdi_combined_flux_err/HD19467_flux_MJy["F444W"]
    rdi_contrast_5sig_combined[np.where(~np.isfinite(rdi_contrast_5sig_combined))] = np.nan
    rdi_contrast_5sig_1D_seps = np.arange(0,3,0.1)
    rdi_contrast_5sig_1D_med = np.zeros(rdi_contrast_5sig_1D_seps.shape)+np.nan
    rdi_contrast_5sig_1D_min = np.zeros(rdi_contrast_5sig_1D_seps.shape)+np.nan
    for k,sep in enumerate(rdi_contrast_5sig_1D_seps):
        rdi_contrast_5sig_1D_med[k] = np.nanmedian(rdi_contrast_5sig_combined[np.where((rdi_rs_grid>(sep-0.05))*(rdi_rs_grid<(sep+0.05)))])
        rdi_contrast_5sig_1D_min[k] = np.nanmin(rdi_contrast_5sig_combined[np.where((rdi_rs_grid>(sep-0.05))*(rdi_rs_grid<(sep+0.05)))])



    # plt.show()
    fontsize=12
    if 1: #figure 2
        fig = plt.figure(1, figsize=(12,5))
        gs = gridspec.GridSpec(2,5, height_ratios=[0.05,1], width_ratios=[0.2,1,1,0.3,1])
        gs.update(left=0.05, right=0.95, bottom=0.19, top=0.85, wspace=0.0, hspace=0.0)

        fontsize=12
        ax1 = plt.subplot(gs[1, 1])
        plt1 = plt.imshow(HD19467_median_im/1e-12,origin="lower",extent=[s3d_RA_vec[0]-s3d_dRA/2.,s3d_RA_vec[-1]+s3d_dRA/2.,s3d_Dec_vec[0]-s3d_dDec/2.,s3d_Dec_vec[-1]+s3d_dDec/2.])
        plt.text(0.03, 0.99, "HD 19467 AB\nMedian Cube", fontsize=fontsize, ha='left', va='top', transform=plt.gca().transAxes,color="Black")
        plt.plot(0,0,"*",color="grey",markersize=10)
        txt = plt.text(0,-0.3, 'A', fontsize=fontsize, ha='center', va='top',color="grey") #\n 1.1-2.6 Jy
        txt.set_path_effects([PathEffects.withStroke(linewidth=1, foreground='w')])
        # plt.plot(ra_offset, dec_offset,".",color="white",markersize=10)
        plt.text(ra_offset, dec_offset-0.4, 'B', fontsize=fontsize, ha='center', va='top',color="white")# \n 36-166 $\mu$Jy
        circle = plt.Circle((ra_offset, dec_offset), 0.2, facecolor='#FFFFFF00', edgecolor='white')#,zorder=0
        plt.gca().add_patch(circle)
        plt.xlim([-2.5,2.5])
        plt.xticks([-2,-1,0,1,2])
        plt.ylim([-3.0,2.])
        plt.yticks([-3,-2,-1,0,1,2])
        plt.gca().set_aspect('equal')
        plt.gca().invert_xaxis()

        plt.clim([0,200])
        plt.xlabel("$\Delta$RA (as)",fontsize=fontsize)
        plt.ylabel("$\Delta$Dec (as)",fontsize=fontsize)
        plt.gca().tick_params(axis='x', labelsize=fontsize)
        plt.gca().tick_params(axis='y', labelsize=fontsize)

        cbax = plt.subplot(gs[0, 1])
        cb = Colorbar(ax=cbax, mappable=plt1, orientation='horizontal', ticklocation='top')
        cb.set_label(r'Flux ($\mu$Jy)', labelpad=5, fontsize=fontsize)
        cb.set_ticks([0, 50,100,150,200])  # xticks[1::]
        plt.gca().tick_params(axis='x', labelsize=fontsize)
        plt.gca().tick_params(axis='y', labelsize=fontsize)

        # plt.subplot(1,3,2)
        ax1 = plt.subplot(gs[1, 2])
        kmax,lmax = np.unravel_index(np.nanargmax(combined_SNR),combined_flux.shape)
        plt2 = plt.imshow(combined_SNR,origin="lower",extent=[nrs2_RA_vec[0]-dra/2.,nrs2_RA_vec[-1]+dra/2.,nrs2_Dec_vec[0]-ddec/2.,nrs2_Dec_vec[-1]+ddec/2.])
        plt.text(0.03, 0.99, "Companion S/N", fontsize=fontsize, ha='left', va='top', transform=plt.gca().transAxes,color="Black")
        # plt.text(nrs2_RA_vec[lmax]+0.1, nrs2_Dec_vec[kmax]+0.3, "S/N={0:.0f}".format(combined_SNR[kmax,lmax]), fontsize=fontsize, ha='center', va='bottom',color="white") #, transform=plt.gca().transAxes
        plt.plot(0,0,"*",color="grey",markersize=10)
        txt = plt.text(0,-0.3, 'A', fontsize=fontsize, ha='center', va='top',color="grey") #\n 1.1-2.6 Jy
        txt.set_path_effects([PathEffects.withStroke(linewidth=1, foreground='w')])
        # plt.plot(ra_offset, dec_offset,".",color="white",markersize=10)
        plt.text(ra_offset, dec_offset-0.4, 'B', fontsize=fontsize, ha='center', va='top',color="white")# \n 36-166 $\mu$Jy
        circle = plt.Circle((ra_offset, dec_offset), 0.2, facecolor='#FFFFFF00', edgecolor='white')#,zorder=0
        plt.gca().add_patch(circle)
        if 1: # plot copass
            plt.text(1.5,-1.95, 'N', fontsize=fontsize, ha='center', va='bottom', color="black")  # \n 1.1-2.6 Jy
            plt.gca().annotate('', xy=(1.5,-2.), xytext=(1.5,-2.5),color="black", fontsize=fontsize,
                arrowprops=dict(facecolor="black",width=1),horizontalalignment='center',verticalalignment='center')
            plt.text(2.05,-2.5, 'E', fontsize=fontsize, ha='right', va='center', color="black")  # \n 1.1-2.6 Jy
            plt.gca().annotate('', xy=(2.0,-2.5), xytext=(1.5,-2.5),color="black", fontsize=fontsize,
                arrowprops=dict(facecolor="black",width=1),horizontalalignment='center',verticalalignment='center')#, shrink=0.001
        # cbar = plt.colorbar(pad=0)#mappable=CS,
        # cbar.ax.tick_params(labelsize=fontsize)
        # cbar.set_label(label='S/N', fontsize=fontsize)
        plt.clim([-2,10])
        # plt.xlim([-rad,rad])
        # plt.ylim([-rad,rad])
        plt.xlim([-2.5,2.5])
        plt.xticks([-2,-1,0,1,2])
        plt.ylim([-3.0,2.])
        plt.yticks([-3,-2,-1,0,1,2])
        plt.gca().set_yticklabels([])
        plt.gca().invert_xaxis()
        plt.gca().set_aspect('equal')
        plt.xlabel("$\Delta$RA (as)",fontsize=fontsize)
        plt.gca().tick_params(axis='x', labelsize=fontsize)

        cbax = plt.subplot(gs[0, 2])
        cb = Colorbar(ax=cbax, mappable=plt2, orientation='horizontal', ticklocation='top')
        cb.set_label(r'S/N', labelpad=5, fontsize=fontsize)
        cb.set_ticks([0,2,4,6,8,10])  # xticks[1::]
        plt.gca().tick_params(axis='x', labelsize=fontsize)
        plt.gca().tick_params(axis='y', labelsize=fontsize)

        ax1 = plt.subplot(gs[1, 4])
        plt.scatter(rgrid_psfprofile.ravel(),im_psfprofile.ravel(),c="grey",s=20,label="PSF profile",marker="+",linewidths=1,alpha=1)
        plt.scatter(rs_grid,contrast_5sig_combined,s=0.2,c=color_list[0],alpha=0.2)
        plt.plot(contrast_5sig_1D_seps,contrast_5sig_1D_med,label="5$\sigma$ FM Median",alpha=1,c=color_list[0],linestyle="-")
        plt.scatter(rdi_rs_grid,rdi_contrast_5sig_combined,s=0.2,c=color_list[1],alpha=0.2)
        plt.plot(rdi_contrast_5sig_1D_seps,rdi_contrast_5sig_1D_med,label="5$\sigma$ RDI Median",alpha=1,c=color_list[1],linestyle="-.")
        plt.plot(F444W_Carter_seps,F444W_Carter_cons,alpha=1,c="black",linestyle="-")#,label="NIRCam ADI+RDI"
        plt.text(3,1e-6, 'NIRCam', fontsize=fontsize, ha='left', va='bottom',color="black",rotation=0)# \n 36-166 $\mu$Jy
        plt.fill_between([0,0.3],[10**(-10),10**(-10)],[1,1],color="grey",alpha=0.3)

        plt.plot(np.sqrt(ra_offset**2+dec_offset**2),HD19467B_NIRSpec_flux_MJy["F444W"]/HD19467_flux_MJy["F444W"],".",color="black",markersize=10)
        plt.text(np.sqrt(ra_offset**2+dec_offset**2)+0.1,HD19467B_NIRSpec_flux_MJy["F444W"]/HD19467_flux_MJy["F444W"], 'HD19467B', fontsize=fontsize, ha='left', va='center',color="black")# \n 36-166 $\mu$Jy

        plt.yscale("log")
        plt.xscale("log")
        # plt.xlim([0,3])
        plt.xlim([0.1,10])
        plt.ylim([10**(-6.3),10**(-1)])
        plt.xlabel("Separation (as)",fontsize=fontsize)
        plt.ylabel("Flux ratio ({0})".format("F444W"),fontsize=fontsize)
        plt.legend(loc="upper right",frameon=True,fontsize=10)
        plt.gca().tick_params(axis='x', labelsize=fontsize)
        plt.gca().tick_params(axis='y', labelsize=fontsize)




        plt.tight_layout()
        out_filename = os.path.join(out_png, "summarydetec.png")
        print("Saving " + out_filename)
        plt.savefig(out_filename, dpi=300)
        plt.savefig(out_filename.replace(".png", ".pdf"))

        out_filename = os.path.join(out_png, "summarydetec_hd.png")
        print("Saving " + out_filename)
        plt.savefig(out_filename, dpi=600)
        plt.savefig(out_filename.replace(".png", ".pdf"))

    if 1:
        fig = plt.figure(2, figsize=(12,4))
        gs = gridspec.GridSpec(2,6, height_ratios=[0.05,1], width_ratios=[0.2,1,1,1,0.35,1])
        gs.update(left=0.05, right=0.95, bottom=0.19, top=0.85, wspace=0.0, hspace=0.0)
        # gs.update(left=0.00, right=1, bottom=0.0, top=1, wspace=0.0, hspace=0.0)
        ax1 = plt.subplot(gs[1, 1])
        kmax,lmax = np.unravel_index(np.nanargmax(combined_SNR),combined_flux.shape)
        plt1 = plt.imshow(combined_flux/1e-12,origin="lower",extent=[nrs2_RA_vec[0]-dra/2.,nrs2_RA_vec[-1]+dra/2.,nrs2_Dec_vec[0]-ddec/2.,nrs2_Dec_vec[-1]+ddec/2.])
        plt.text(0.03, 0.99, "Flux (F444W)", fontsize=fontsize, ha='left', va='top', transform=plt.gca().transAxes,color="Black")
        plt.text(0.03, 0.01, "Forward Model (FM)", fontsize=1.4*fontsize, ha='left', va='bottom', transform=plt.gca().transAxes,color=color_list[0],zorder=10)
        # plt.text(nrs2_RA_vec[lmax]+0.1, nrs2_Dec_vec[kmax]+0.3, "S/N={0:.0f}".format(combined_SNR[kmax,lmax]), fontsize=fontsize, ha='center', va='bottom',color="white") #, transform=plt.gca().transAxes
        plt.plot(0,0,"*",color="grey",markersize=10)
        plt.text(0,-0.3, 'A', fontsize=fontsize, ha='center', va='top',color="grey") #\n 1.1-2.6 Jy
        # plt.plot(ra_offset, dec_offset,".",color="white",markersize=10)
        plt.text(ra_offset, dec_offset-0.4, 'B', fontsize=fontsize, ha='center', va='top',color="white")# \n 36-166 $\mu$Jy
        circle = plt.Circle((ra_offset, dec_offset), 0.2, facecolor='#FFFFFF00', edgecolor='white')#,zorder=0
        plt.gca().add_patch(circle)
        # plt.clim([-5,5])
        plt.clim([-50,50])
        plt.xlim([-2.5,2.5])
        plt.xticks([-2,-1,0,1,2])
        plt.ylim([-3.0,2.])
        plt.yticks([-3,-2,-1,0,1,2])
        plt.gca().invert_xaxis()
        plt.gca().set_aspect('equal')
        plt.xlabel("$\Delta$RA (as)",fontsize=fontsize)
        plt.ylabel("$\Delta$Dec (as)",fontsize=fontsize)
        # plt.yticks([])
        plt.gca().tick_params(axis='x', labelsize=fontsize)
        plt.gca().tick_params(axis='y', labelsize=fontsize)

        cbax = plt.subplot(gs[0, 1])
        cb = Colorbar(ax=cbax, mappable=plt1, orientation='horizontal', ticklocation='top')
        cb.set_label(r'Flux ($\mu$Jy)', labelpad=5, fontsize=fontsize)
        # cb.set_ticks([-5,-2.5,0,2.5])  # xticks[1::]
        cb.set_ticks([-50,-25,0,25])  # xticks[1::]
        plt.gca().tick_params(axis='x', labelsize=fontsize)
        plt.gca().tick_params(axis='y', labelsize=fontsize)

        ax2 = plt.subplot(gs[1, 2])
        kmax,lmax = np.unravel_index(np.nanargmax(combined_SNR),combined_flux.shape)
        plt2 = plt.imshow(np.log10(combined_flux_err/1e-12),origin="lower",extent=[nrs2_RA_vec[0]-dra/2.,nrs2_RA_vec[-1]+dra/2.,nrs2_Dec_vec[0]-ddec/2.,nrs2_Dec_vec[-1]+ddec/2.])
        plt.text(0.03, 0.99, "Flux error (F444W)", fontsize=fontsize, ha='left', va='top', transform=plt.gca().transAxes,color="Black")
        # plt.text(nrs2_RA_vec[lmax]+0.1, nrs2_Dec_vec[kmax]+0.3, "S/N={0:.0f}".format(combined_SNR[kmax,lmax]), fontsize=fontsize, ha='center', va='bottom',color="white") #, transform=plt.gca().transAxes
        plt.plot(0,0,"*",color="grey",markersize=10)
        plt.text(0,-0.3, 'A', fontsize=fontsize, ha='center', va='top',color="grey") #\n 1.1-2.6 Jy
        # plt.plot(ra_offset, dec_offset,".",color="white",markersize=10)
        circle = plt.Circle((ra_offset, dec_offset), 0.2, facecolor='#FFFFFF00', edgecolor='white')#,zorder=0
        plt.gca().add_patch(circle)
        plt.text(ra_offset, dec_offset-0.4, 'B', fontsize=fontsize, ha='center', va='top',color="white")# \n 36-166 $\mu$Jy
        plt.clim([-1.0,1.0])
        plt.xlim([-2.5,2.5])
        plt.xticks([-2,-1,0,1,2])
        plt.ylim([-3.0,2.])
        plt.yticks([-3,-2,-1,0,1,2])
        plt.gca().set_yticklabels([])
        plt.gca().invert_xaxis()
        plt.gca().set_aspect('equal')
        plt.xlabel("$\Delta$RA (as)",fontsize=fontsize)
        # plt.ylabel("$\Delta$Dec (as)",fontsize=fontsize)
        # plt.yticks([])
        plt.gca().tick_params(axis='x', labelsize=fontsize)
        # plt.gca().tick_params(axis='y', labelsize=fontsize)

        cbax = plt.subplot(gs[0, 2])
        cb = Colorbar(ax=cbax, mappable=plt2, orientation='horizontal', ticklocation='top')
        cb.set_label(r'Flux ($\mu$Jy)', labelpad=5, fontsize=fontsize)
        cb.set_ticks([-1, 0,1])
        cb.set_ticklabels(["$10^{-1}$","$10^{0}$","$10^{1}$"])  # xticks[1::]
        # cb.set_ticklabels(["0.1","1","10"])  # xticks[1::]
        plt.gca().tick_params(axis='x', labelsize=fontsize)
        plt.gca().tick_params(axis='y', labelsize=fontsize)

        ax3 = plt.subplot(gs[1, 3])
        kmax,lmax = np.unravel_index(np.nanargmax(combined_SNR),combined_flux.shape)
        plt2 = plt.imshow(combined_SNR,origin="lower",extent=[nrs2_RA_vec[0]-dra/2.,nrs2_RA_vec[-1]+dra/2.,nrs2_Dec_vec[0]-ddec/2.,nrs2_Dec_vec[-1]+ddec/2.])
        plt.text(0.03, 0.99, "Companion S/N", fontsize=fontsize, ha='left', va='top', transform=plt.gca().transAxes,color="Black")
        # plt.text(nrs2_RA_vec[lmax]+0.1, nrs2_Dec_vec[kmax]+0.3, "S/N={0:.0f}".format(combined_SNR[kmax,lmax]), fontsize=fontsize, ha='center', va='bottom',color="white") #, transform=plt.gca().transAxes
        plt.plot(0,0,"*",color="grey",markersize=10)
        plt.text(0,-0.3, 'A', fontsize=fontsize, ha='center', va='top',color="grey") #\n 1.1-2.6 Jy
        # plt.plot(ra_offset, dec_offset,".",color="white",markersize=10)
        circle = plt.Circle((ra_offset, dec_offset), 0.2, facecolor='#FFFFFF00', edgecolor='white')#,zorder=0
        plt.gca().add_patch(circle)
        plt.text(ra_offset, dec_offset-0.4, 'B', fontsize=fontsize, ha='center', va='top',color="white")# \n 36-166 $\mu$Jy
        plt.clim([-2,10])
        plt.xlim([-2.5,2.5])
        plt.xticks([-2,-1,0,1,2])
        plt.ylim([-3.0,2.])
        plt.yticks([-3,-2,-1,0,1,2])
        plt.gca().set_yticklabels([])
        plt.gca().invert_xaxis()
        plt.gca().set_aspect('equal')
        plt.xlabel("$\Delta$RA (as)",fontsize=fontsize)
        # plt.ylabel("$\Delta$Dec (as)",fontsize=fontsize)
        # plt.yticks([])
        plt.gca().tick_params(axis='x', labelsize=fontsize)
        # plt.gca().tick_params(axis='y', labelsize=fontsize)

        cbax = plt.subplot(gs[0, 3])
        cb = Colorbar(ax=cbax, mappable=plt2, orientation='horizontal', ticklocation='top')
        cb.set_label(r'S/N', labelpad=5, fontsize=fontsize)
        cb.set_ticks([0,2,4,6,8,10])  # xticks[1::]
        plt.gca().tick_params(axis='x', labelsize=fontsize)
        plt.gca().tick_params(axis='y', labelsize=fontsize)

        ax2 = plt.subplot(gs[0:2, 5])
        snr_map_masked = copy(combined_SNR)
        nan_mask_boxsize=5
        snr_map_masked[np.where(np.isnan(correlate2d(snr_map_masked,np.ones((nan_mask_boxsize,nan_mask_boxsize)),mode="same")))] = np.nan
        snr_map_masked[np.where((fm_r2comp_grid < 0.7))] = np.nan
        # plt.figure(10)
        # plt.imshow(snr_map_masked)
        # plt.show()
        hist, bins = np.histogram(snr_map_masked[np.where(np.isfinite(snr_map_masked))], bins=20 * 3,range=(-10, 10))
        bin_centers = (bins[1::] + bins[0:np.size(bins) - 1]) / 2.
        plt.plot(bin_centers, hist / (np.nansum(hist) * (bins[1] - bins[0])), label="Combined",color=color_list[0],linestyle="-")
        plt.plot(nrs1_bins,nrs1_hist, label="NRS1",color=color_list[1],linestyle=":")
        plt.plot(nrs2_bins,nrs2_hist, label="NRS2",color="pink",linestyle="-.")

        plt.plot(bin_centers, 1 / np.sqrt(2 * np.pi) * np.exp(-0.5 * (bin_centers - 0.0) ** 2), color="black",
                 linestyle="--", label="Gaussian")
        plt.yscale("log")
        plt.xlim([-5, 5])
        plt.ylim([1e-4, 1])
        plt.xlabel("SNR",fontsize=fontsize)
        plt.ylabel("PDF",fontsize=fontsize)
        plt.legend()
        plt.gca().tick_params(axis='x', labelsize=fontsize)
        plt.gca().tick_params(axis='y', labelsize=fontsize)

        out_filename = os.path.join(out_png, "FMdetec_summary.png")
        print("Saving " + out_filename)
        plt.savefig(out_filename, dpi=300)
        plt.savefig(out_filename.replace(".png", ".pdf"))

        out_filename = os.path.join(out_png, "FMdetec_summary_hd.png")
        print("Saving " + out_filename)
        plt.savefig(out_filename, dpi=600)
        plt.savefig(out_filename.replace(".png", ".pdf"))

    if 1:
        fig = plt.figure(3, figsize=(12,4))
        gs = gridspec.GridSpec(2,6, height_ratios=[0.05,1], width_ratios=[0.2,1,1,1,0.35,1])
        gs.update(left=0.05, right=0.95, bottom=0.19, top=0.85, wspace=0.0, hspace=0.0)
        # gs.update(left=0.00, right=1, bottom=0.0, top=1, wspace=0.0, hspace=0.0)
        ax1 = plt.subplot(gs[1, 1])
        plt1 = plt.imshow(rdi_combined_flux/1e-12,origin="lower",extent=[rdi_nrs1_RA_vec[0]-rdi_dra/2.,rdi_nrs1_RA_vec[-1]+rdi_dra/2.,rdi_nrs1_Dec_vec[0]-rdi_ddec/2.,rdi_nrs1_Dec_vec[-1]+rdi_ddec/2.])
        plt.text(0.03, 0.99, "Flux (F444W)", fontsize=fontsize, ha='left', va='top', transform=plt.gca().transAxes,color="Black")
        plt.text(0.03, 0.01, "RDI", fontsize=1.5*fontsize, ha='left', va='bottom', transform=plt.gca().transAxes,color=color_list[1])
        plt.plot(0,0,"*",color="grey",markersize=10)
        plt.text(0,-0.3, 'A', fontsize=fontsize, ha='center', va='top',color="grey") #\n 1.1-2.6 Jy
        # plt.plot(ra_offset, dec_offset,".",color="white",markersize=10)
        circle = plt.Circle((ra_offset, dec_offset), 0.2, facecolor='#FFFFFF00', edgecolor='white')#,zorder=0
        plt.gca().add_patch(circle)
        plt.text(ra_offset, dec_offset-0.4, 'B', fontsize=fontsize, ha='center', va='top',color="white")# \n 36-166 $\mu$Jy
        plt.clim([-50,50])
        plt.xlim([-2.5,2.5])
        plt.xticks([-2,-1,0,1,2])
        plt.ylim([-3.0,2.])
        plt.yticks([-3,-2,-1,0,1,2])
        plt.gca().invert_xaxis()
        plt.gca().set_aspect('equal')
        plt.xlabel("$\Delta$RA (as)",fontsize=fontsize)
        plt.ylabel("$\Delta$Dec (as)",fontsize=fontsize)
        # plt.yticks([])
        plt.gca().tick_params(axis='x', labelsize=fontsize)
        plt.gca().tick_params(axis='y', labelsize=fontsize)

        cbax = plt.subplot(gs[0, 1])
        cb = Colorbar(ax=cbax, mappable=plt1, orientation='horizontal', ticklocation='top')
        cb.set_label(r'Flux ($\mu$Jy)', labelpad=5, fontsize=fontsize)
        cb.set_ticks([-50,-25,0,25])  # xticks[1::]
        plt.gca().tick_params(axis='x', labelsize=fontsize)
        plt.gca().tick_params(axis='y', labelsize=fontsize)

        ax2 = plt.subplot(gs[1, 2])
        plt2 = plt.imshow(np.log10(rdi_combined_flux_err/1e-12),origin="lower",extent=[rdi_nrs1_RA_vec[0]-rdi_dra/2.,rdi_nrs1_RA_vec[-1]+rdi_dra/2.,rdi_nrs1_Dec_vec[0]-rdi_ddec/2.,rdi_nrs1_Dec_vec[-1]+rdi_ddec/2.])
        plt.text(0.03, 0.99, "Flux error (F444W)", fontsize=fontsize, ha='left', va='top', transform=plt.gca().transAxes,color="Black")
        plt.plot(0,0,"*",color="grey",markersize=10)
        plt.text(0,-0.3, 'A', fontsize=fontsize, ha='center', va='top',color="grey") #\n 1.1-2.6 Jy
        # plt.plot(ra_offset, dec_offset,".",color="white",markersize=10)
        circle = plt.Circle((ra_offset, dec_offset), 0.2, facecolor='#FFFFFF00', edgecolor='white')#,zorder=0
        plt.gca().add_patch(circle)
        plt.text(ra_offset, dec_offset-0.4, 'B', fontsize=fontsize, ha='center', va='top',color="white")# \n 36-166 $\mu$Jy
        plt.clim([1.0,3.0])
        plt.xlim([-2.5,2.5])
        plt.xticks([-2,-1,0,1,2])
        plt.ylim([-3.0,2.])
        plt.yticks([-3,-2,-1,0,1,2])
        plt.gca().set_yticklabels([])
        plt.gca().invert_xaxis()
        plt.gca().set_aspect('equal')
        plt.xlabel("$\Delta$RA (as)",fontsize=fontsize)
        # plt.ylabel("$\Delta$Dec (as)",fontsize=fontsize)
        # plt.yticks([])
        plt.gca().tick_params(axis='x', labelsize=fontsize)
        # plt.gca().tick_params(axis='y', labelsize=fontsize)

        cbax = plt.subplot(gs[0, 2])
        cb = Colorbar(ax=cbax, mappable=plt2, orientation='horizontal', ticklocation='top')
        cb.set_label(r'Flux ($\mu$Jy)', labelpad=5, fontsize=fontsize)
        cb.set_ticks([1, 2,3])
        cb.set_ticklabels(["$10^{1}$","$10^{2}$","$10^{3}$"])  # xticks[1::]
        # cb.set_ticklabels(["0.1","1","10"])  # xticks[1::]
        plt.gca().tick_params(axis='x', labelsize=fontsize)
        plt.gca().tick_params(axis='y', labelsize=fontsize)

        ax3 = plt.subplot(gs[1, 3])
        # rdi_combined_SNR[np.where((rdi_r2star_grid < 0.5))] = np.nan
        plt2 = plt.imshow(rdi_combined_SNR,origin="lower",extent=[rdi_nrs1_RA_vec[0]-rdi_dra/2.,rdi_nrs1_RA_vec[-1]+rdi_dra/2.,rdi_nrs1_Dec_vec[0]-rdi_ddec/2.,rdi_nrs1_Dec_vec[-1]+rdi_ddec/2.])
        plt.text(0.03, 0.99, "Companion S/N", fontsize=fontsize, ha='left', va='top', transform=plt.gca().transAxes,color="Black")
        plt.plot(0,0,"*",color="grey",markersize=10)
        plt.text(0,-0.3, 'A', fontsize=fontsize, ha='center', va='top',color="grey") #\n 1.1-2.6 Jy
        # plt.plot(ra_offset, dec_offset,".",color="white",markersize=10)
        circle = plt.Circle((ra_offset, dec_offset), 0.2, facecolor='#FFFFFF00', edgecolor='white')#,zorder=0
        plt.gca().add_patch(circle)
        plt.text(ra_offset, dec_offset-0.4, 'B', fontsize=fontsize, ha='center', va='top',color="white")# \n 36-166 $\mu$Jy
        plt.clim([-2,10])
        plt.xlim([-2.5,2.5])
        plt.xticks([-2,-1,0,1,2])
        plt.ylim([-3.0,2.])
        plt.yticks([-3,-2,-1,0,1,2])
        plt.gca().set_yticklabels([])
        plt.gca().invert_xaxis()
        plt.gca().set_aspect('equal')
        plt.xlabel("$\Delta$RA (as)",fontsize=fontsize)
        # plt.ylabel("$\Delta$Dec (as)",fontsize=fontsize)
        # plt.yticks([])
        plt.gca().tick_params(axis='x', labelsize=fontsize)
        # plt.gca().tick_params(axis='y', labelsize=fontsize)

        cbax = plt.subplot(gs[0, 3])
        cb = Colorbar(ax=cbax, mappable=plt2, orientation='horizontal', ticklocation='top')
        cb.set_label(r'S/N', labelpad=5, fontsize=fontsize)
        cb.set_ticks([0,2,4,6,8,10])  # xticks[1::]
        plt.gca().tick_params(axis='x', labelsize=fontsize)
        plt.gca().tick_params(axis='y', labelsize=fontsize)

        ax2 = plt.subplot(gs[0:2, 5])
        snr_map_masked = copy(rdi_combined_SNR)
        nan_mask_boxsize=5
        snr_map_masked[np.where(np.isnan(correlate2d(snr_map_masked,np.ones((nan_mask_boxsize,nan_mask_boxsize)),mode="same")))] = np.nan
        snr_map_masked[np.where((rdi_r2comp_grid < 0.4))] = np.nan
        # snr_map_masked[np.where((rdi_r2star_grid < 0.5))] = np.nan
        # plt.figure(10)
        # plt.imshow(snr_map_masked)
        # plt.show()
        hist, bins = np.histogram(snr_map_masked[np.where(np.isfinite(snr_map_masked))], bins=20 * 3,range=(-10, 10))
        bin_centers = (bins[1::] + bins[0:np.size(bins) - 1]) / 2.
        plt.plot(bin_centers, hist / (np.nansum(hist) * (bins[1] - bins[0])), label="Combined",color=color_list[0],linestyle="-")
        # plt.plot(nrs1_bins,nrs1_hist, label="NRS1",color=color_list[1],linestyle=":")
        # plt.plot(nrs2_bins,nrs2_hist, label="NRS2",color="pink",linestyle="-.")

        plt.plot(bin_centers, 1 / np.sqrt(2 * np.pi) * np.exp(-0.5 * (bin_centers - 0.0) ** 2), color="black",
                 linestyle="--", label="Gaussian")
        plt.yscale("log")
        plt.xlim([-5, 5])
        plt.ylim([1e-4, 1])
        plt.xlabel("SNR",fontsize=fontsize)
        plt.ylabel("PDF",fontsize=fontsize)
        plt.legend()
        plt.gca().tick_params(axis='x', labelsize=fontsize)
        plt.gca().tick_params(axis='y', labelsize=fontsize)

        out_filename = os.path.join(out_png, "RDIdetec_summary.png")
        print("Saving " + out_filename)
        plt.savefig(out_filename, dpi=300)
        plt.savefig(out_filename.replace(".png", ".pdf"))

        out_filename = os.path.join(out_png, "RDIdetec_summary_hd.png")
        print("Saving " + out_filename)
        plt.savefig(out_filename, dpi=600)
        plt.savefig(out_filename.replace(".png", ".pdf"))

    if 1:
        # webbpsf_filename = "/stow/jruffio/data/JWST/nirspec/HD_19467/breads/20230626_utils/jw01414004001_02101_nrs1_webbpsf.fits"
        # print("loading" , webbpsf_filename)
        # with fits.open(webbpsf_filename) as hdulist:
        #     wpsfs1 = hdulist[0].data
        #     wpsfs_header = hdulist[0].header
        #     wepsfs1 = hdulist[1].data
        #     webbpsf1_wvs = hdulist[2].data
        #     webbpsf_X = hdulist[3].data
        #     webbpsf_Y = hdulist[4].data
        #     wpsf_pixelscale = wpsfs_header["PIXELSCL"]
        #     wpsf_oversample = wpsfs_header["oversamp"]
        # arg3um = np.argmin(np.abs(webbpsf1_wvs-3.0))
        # webbpsf_filename = "/stow/jruffio/data/JWST/nirspec/HD_19467/breads/20230626_utils/jw01414004001_02101_nrs2_webbpsf.fits"
        # print("loading" , webbpsf_filename)
        # with fits.open(webbpsf_filename) as hdulist:
        #     wpsfs2 = hdulist[0].data
        #     wpsfs_header = hdulist[0].header
        #     wepsfs2 = hdulist[1].data
        #     webbpsf2_wvs = hdulist[2].data
        #     webbpsf_X = hdulist[3].data
        #     webbpsf_Y = hdulist[4].data
        #     wpsf_pixelscale = wpsfs_header["PIXELSCL"]
        #     wpsf_oversample = wpsfs_header["oversamp"]
        # arg5um = np.argmin(np.abs(webbpsf2_wvs-5.0))
        # # wpsf_profile  = (np.nansum(wepsfs1,axis=0)+np.nansum(wepsfs2,axis=0))/2.
        # # print(wpsf_profile.shape)
        # webbpsf_R = np.sqrt(webbpsf_X ** 2 + webbpsf_Y ** 2)
        # print(webbpsf_R.shape)

        fig = plt.figure(4, figsize=(6,5))
        plt.scatter(rgrid_psfprofile.ravel(),im_psfprofile.ravel(),c="grey",s=20,label="PSF profile",marker="+",linewidths=1,alpha=1)
        # plt.scatter(webbpsf_R,wepsfs1[arg3um,:,:]/np.nanmax(wepsfs1[arg3um,:,:]),s=0.2,label="WebbPSF (3$\mu$m)",c="grey",alpha=0.3,marker="x")
        # plt.scatter(webbpsf_R,wepsfs2[arg5um,:,:]/np.nanmax(wepsfs2[arg5um,:,:]),s=0.2,label="WebbPSF (5$\mu$m)",c="grey",alpha=0.3,marker="+")
        # plt.scatter(webbpsf_R,wpsfs1[arg3um,:,:]/np.nanmax(wpsfs1[arg3um,:,:]),s=0.2,label="WebbPSF (3$\mu$m)",c="red",alpha=0.3,marker="x")
        # plt.scatter(webbpsf_R,wpsfs2[arg5um,:,:]/np.nanmax(wpsfs2[arg5um,:,:]),s=0.2,label="WebbPSF (5$\mu$m)",c="red",alpha=0.3,marker="+")
        plt.scatter(rs_grid,contrast_5sig_combined,s=0.2,c=color_list[0],alpha=0.3)#,label="5$\sigma$ - Forward Model"
        plt.plot(contrast_5sig_1D_seps,contrast_5sig_1D_med,label="5$\sigma$ - Forward Model - Median",alpha=1,c=color_list[0],linestyle="-")
        # plt.plot(contrast_5sig_1D_seps,contrast_5sig_1D_min,label="5$\sigma$ - Forward Model - Best",alpha=1,c=color_list[0],linestyle="--")
        plt.scatter(rdi_rs_grid,rdi_contrast_5sig_combined,s=0.2,c=color_list[1],alpha=0.3)#,label="5$\sigma$ - RDI"
        plt.plot(rdi_contrast_5sig_1D_seps,rdi_contrast_5sig_1D_med,label="5$\sigma$ - RDI - Median",alpha=1,c=color_list[1],linestyle="-.")
        # plt.plot(rdi_contrast_5sig_1D_seps,rdi_contrast_5sig_1D_min,label="5$\sigma$ - RDI - Best",alpha=1,c=color_list[1],linestyle=":")
        plt.plot(F444W_Carter_seps,F444W_Carter_cons*np.sqrt(1236/2100),alpha=1,c="black",linestyle="-.")#,label="NIRCam ADI+RDI"
        plt.text(1.2,4e-6, 'NIRCam Carter+2023', fontsize=fontsize, ha='left', va='center',color="black",rotation=-11)# \n 36-166 $\mu$Jy
        plt.fill_between([0,0.3],[10**(-10),10**(-10)],[1,1],color="grey",alpha=0.2)

        plt.plot(np.sqrt(ra_offset**2+dec_offset**2),HD19467B_NIRSpec_flux_MJy["F444W"]/HD19467_flux_MJy["F444W"],".",color="black",markersize=10)
        plt.text(np.sqrt(ra_offset**2+dec_offset**2)+0.1,HD19467B_NIRSpec_flux_MJy["F444W"]/HD19467_flux_MJy["F444W"], 'HD19467B', fontsize=fontsize, ha='left', va='center',color="black")# \n 36-166 $\mu$Jy

        plt.yscale("log")
        plt.xscale("log")
        # plt.xlim([0,3])
        plt.xlim([0.1,10])
        plt.ylim([10**(-6.3),10**(-0)])
        plt.xlabel("Separation (as)",fontsize=fontsize)
        plt.ylabel("Flux ratio ({0})".format("F444W"),fontsize=fontsize)
        plt.legend(loc="upper right",frameon=True,fontsize=10)
        plt.gca().tick_params(axis='x', labelsize=fontsize)
        plt.gca().tick_params(axis='y', labelsize=fontsize)

        out_filename = os.path.join(out_png, "contrasts.png")
        print("Saving " + out_filename)
        plt.savefig(out_filename, dpi=300)
        plt.savefig(out_filename.replace(".png", ".pdf"))

        out_filename = os.path.join(out_png, "contrasts_hd.png")
        print("Saving " + out_filename)
        plt.savefig(out_filename, dpi=600)
        plt.savefig(out_filename.replace(".png", ".pdf"))

    plt.show()
