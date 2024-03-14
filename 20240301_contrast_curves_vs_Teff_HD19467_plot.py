
import os
import numpy as np
import matplotlib.pyplot as plt
from  scipy.interpolate import interp1d
import astropy.io.fits as fits
from glob import glob
import h5py
from scipy.signal import correlate2d
from scipy.interpolate import RegularGridInterpolator
import astropy.units as u
from astropy import constants as const

from copy import copy
# import matplotlib
# matplotlib.use('Qt5Agg')


import matplotlib.gridspec as gridspec # GRIDSPEC !
from matplotlib.colorbar import Colorbar

if __name__ == "__main__":
    try:
        import mkl
        mkl.set_num_threads(1)
    except:
        pass


    ####################
    ## To be modified
    ####################
    teffs_list = [500,1000, 1500,2000,2500,3000]
    # teffs_list = [500,1000]
    # teffs_list = [ 1500,2000,2500,3000]
    color_list = ["#ff9900","#006699", "#6600ff", "#006699", "#ff9900", "#6600ff"]
    linelist_list = ["-","--", "-.", ":",  (0, (3, 10, 1, 10)), (0, (5, 10))]
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
    out_dir0 = "/stow/jruffio/data/JWST/nirspec/HD_19467/breads/20240216_out_fm/"
    if not os.path.exists(out_png):
        os.makedirs(out_png)
    # Science data: List of stage 2 cal.fits files
    filelist = glob("/stow/jruffio/data/JWST/nirspec/HD_19467/HD19467_onaxis_roll2/20240124_stage2_clean/jw01414004001_02101_*_nrs*_cal.fits")
    filelist.sort()
    for filename in filelist:
        print(filename)
    print("N files: {0}".format(len(filelist)))
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
    ####################






    if 0:
        for myteff in teffs_list:
            out_dir = os.path.join(out_dir0,"xy_{0}K/".format(myteff))
            for detector, photfilter_name in zip(["nrs2","nrs1"],["F460M","F360M"]):

                outoftheoven_filelist = []
                for filename in filelist:
                    if detector not in filename:
                        continue
                    print(os.path.join(out_dir, "*_" + os.path.basename(filename).replace(".fits", "_out.fits")))
                    tmp_filelist = glob(
                        os.path.join(out_dir, "*_" + os.path.basename(filename).replace(".fits", "_out.fits")))
                    tmp_filelist.sort()
                    if len(tmp_filelist) == 0:
                        continue
                    outoftheoven_filelist.append(tmp_filelist[-1])

                N_files = len(outoftheoven_filelist)
                fluxmap_list = []
                fluxerrmap_list = []
                snr_vals = []
                for fid,outoftheoven_filename in enumerate(outoftheoven_filelist):
                    print(outoftheoven_filename)
                    with h5py.File(outoftheoven_filename, 'r') as hf:
                        rvs = np.array(hf.get("rvs"))
                        ras = np.array(hf.get("ras"))
                        decs = np.array(hf.get("decs"))
                        log_prob = np.array(hf.get("log_prob"))
                        log_prob_H0 = np.array(hf.get("log_prob_H0"))
                        rchi2 = np.array(hf.get("rchi2"))
                        linparas = np.array(hf.get("linparas"))
                        linparas_err = np.array(hf.get("linparas_err"))
                        print(ras.shape)

                    k = 0

                    linparas = np.swapaxes(linparas, 1, 2)
                    linparas_err = np.swapaxes(linparas_err, 1, 2)
                    log_prob = np.swapaxes(log_prob, 1, 2)
                    log_prob_H0 = np.swapaxes(log_prob_H0, 1, 2)
                    rchi2 = np.swapaxes(rchi2, 1, 2)

                    dra=ras[1]-ras[0]
                    ddec=decs[1]-decs[0]
                    ras_grid,decs_grid = np.meshgrid(ras,decs)
                    rs_grid = np.sqrt(ras_grid**2+decs_grid**2)

                    fluxmap_list.append(linparas[k,:,:,0])
                    fluxerrmap_list.append(linparas_err[k,:,:,0])
                    snr_map = linparas[k,:,:,0]/linparas_err[k,:,:,0]

                    snr_map_masked = copy(snr_map)
                    rs_comp_grid = np.sqrt((ras_grid-ra_offset)**2+(decs_grid-dec_offset)**2)
                    nan_mask_boxsize=5
                    snr_map_masked[np.where(np.isnan(correlate2d(snr_map_masked,np.ones((nan_mask_boxsize,nan_mask_boxsize)),mode="same")))] = np.nan
                    snr_map_masked[np.where((rs_comp_grid < 0.7))] = np.nan
                    snr_vals.append(snr_map_masked[np.where(np.isfinite(snr_map_masked))])
                    hist, bins = np.histogram(snr_map_masked[np.where(np.isfinite(snr_map_masked))], bins=20 * 3,range=(-10, 10))
                    bin_centers = (bins[1::] + bins[0:np.size(bins) - 1]) / 2.

                    contrast_5sig  = 5*linparas_err[k,:,:,0]/HD19467_flux_MJy[photfilter_name]
                    nan_mask_boxsize=2
                    contrast_5sig[np.where(np.isnan(correlate2d(contrast_5sig,np.ones((nan_mask_boxsize,nan_mask_boxsize)),mode="same")))] = np.nan

                fluxmap_arr = np.array(fluxmap_list)
                fluxerrmap_arr = np.array(fluxerrmap_list)
                fluxmap_combined = np.nansum(fluxmap_arr/fluxerrmap_arr**2,axis=0)/np.nansum(1/fluxerrmap_arr**2,axis=0)
                fluxerrmap_combined = 1/np.sqrt(np.nansum(1/fluxerrmap_arr**2,axis=0))
                snr_map_combined = fluxmap_combined/fluxerrmap_combined
                contrast_5sig_combined  = 5*fluxerrmap_combined/HD19467_flux_MJy[photfilter_name]

                hist, bins = np.histogram(np.concatenate(snr_vals), bins=20 * 3, range=(-10, 10))
                bin_centers = (bins[1::] + bins[0:np.size(bins) - 1]) / 2.
                out_filename = os.path.join(out_png,"HD19467B_FM_"+photfilter_name+"_"+detector+"_"+"{0}K".format(myteff)+"_hist.fits")
                print(out_filename)
                hdulist = fits.HDUList()
                hdulist.append(fits.PrimaryHDU(data=bin_centers))
                hdulist.append(fits.ImageHDU(data= hist / (np.nansum(hist) * (bins[1] - bins[0]))))
                try:
                    hdulist.writeto(out_filename, overwrite=True)
                except TypeError:
                    hdulist.writeto(out_filename, clobber=True)
                hdulist.close()

                out_filename = os.path.join(out_png,"HD19467B_FM_"+photfilter_name+"_"+detector+"_"+"{0}K".format(myteff)+"_SNR.fits")
                hdulist = fits.HDUList()
                hdulist.append(fits.PrimaryHDU(data=snr_map_combined))
                try:
                    hdulist.writeto(out_filename, overwrite=True)
                except TypeError:
                    hdulist.writeto(out_filename, clobber=True)
                hdulist.close()

                out_filename = os.path.join(out_png,"HD19467B_FM_"+photfilter_name+"_"+detector+"_"+"{0}K".format(myteff)+"_flux.fits")
                hdulist = fits.HDUList()
                hdulist.append(fits.PrimaryHDU(data=fluxmap_combined))
                try:
                    hdulist.writeto(out_filename, overwrite=True)
                except TypeError:
                    hdulist.writeto(out_filename, clobber=True)
                hdulist.close()

                out_filename = os.path.join(out_png,"HD19467B_FM_"+photfilter_name+"_"+detector+"_"+"{0}K".format(myteff)+"_fluxerr.fits")
                hdulist = fits.HDUList()
                hdulist.append(fits.PrimaryHDU(data=fluxerrmap_combined))
                try:
                    hdulist.writeto(out_filename, overwrite=True)
                except TypeError:
                    hdulist.writeto(out_filename, clobber=True)
                hdulist.close()

                out_filename = os.path.join(out_png,"HD19467B_FM_"+photfilter_name+"_"+detector+"_"+"{0}K".format(myteff)+"_fluxratioerr_1sig.fits")
                hdulist = fits.HDUList()
                hdulist.append(fits.PrimaryHDU(data=contrast_5sig_combined/5.))
                try:
                    hdulist.writeto(out_filename, overwrite=True)
                except TypeError:
                    hdulist.writeto(out_filename, clobber=True)
                hdulist.close()

                out_filename = os.path.join(out_png,"HD19467B_FM_"+photfilter_name+"_"+detector+"_"+"{0}K".format(myteff)+"_RADec.fits")
                hdulist = fits.HDUList()
                hdulist.append(fits.PrimaryHDU(data=ras_grid))
                hdulist.append(fits.ImageHDU(data=decs_grid))
                try:
                    hdulist.writeto(out_filename, overwrite=True)
                except TypeError:
                    hdulist.writeto(out_filename, clobber=True)
                hdulist.close()


                # plt.show()
        exit()

    with h5py.File(os.path.join(external_dir, "BT-Settlchris_0.5-6.0um_Teff500_1600_logg3.5_5.0.hdf5"), 'r') as hf:
        grid_specs = np.array(hf.get("spec"))
        grid_temps = np.array(hf.get("temps"))
        grid_loggs = np.array(hf.get("loggs"))
        grid_wvs = np.array(hf.get("wvs"))
    photfilter = "/stow/jruffio/data/JWST/nirspec/HD_19467/breads/external/JWST_NIRCam." + "F444W" + ".dat"
    filter_arr = np.loadtxt(photfilter)
    trans_wvs = filter_arr[:, 0] / 1e4
    trans = filter_arr[:, 1]
    photfilter_f = interp1d(trans_wvs, trans, bounds_error=False, fill_value=0)
    photfilter_wv0 = np.nansum(trans_wvs * photfilter_f(trans_wvs)) / np.nansum(photfilter_f(trans_wvs))
    grid_dwvs = grid_wvs[1::] - grid_wvs[0:np.size(grid_wvs) - 1]
    grid_dwvs = np.insert(grid_dwvs, 0, grid_dwvs[0])
    filter_norm = np.nansum((grid_dwvs * u.um) * photfilter_f(grid_wvs))
    Flambda = np.nansum((grid_dwvs * u.um)[None, None, :] * photfilter_f(grid_wvs)[None, None, :] * (
                grid_specs * u.W * u.m ** -2 / u.um), axis=2) / filter_norm
    Fnu = Flambda * (photfilter_wv0 * u.um) ** 2 / const.c  # from Flambda back to Fnu
    grid_specs = grid_specs / Fnu[:, :, None].to(u.MJy).value
    myinterpgrid = RegularGridInterpolator((grid_temps, grid_loggs), grid_specs, method="linear",
                                           bounds_error=False, fill_value=np.nan)

    with h5py.File(os.path.join(external_dir, "BT-Settlchris_0.5-6.0um_Teff1500_3000_logg3.5_5.5.hdf5"), 'r') as hf:
        grid_specs = np.array(hf.get("spec"))
        grid_temps = np.array(hf.get("temps"))
        grid_loggs = np.array(hf.get("loggs"))
        grid_wvs = np.array(hf.get("wvs"))
    photfilter = "/stow/jruffio/data/JWST/nirspec/HD_19467/breads/external/JWST_NIRCam." + "F444W" + ".dat"
    filter_arr = np.loadtxt(photfilter)
    trans_wvs = filter_arr[:, 0] / 1e4
    trans = filter_arr[:, 1]
    photfilter_f = interp1d(trans_wvs, trans, bounds_error=False, fill_value=0)
    photfilter_wv0 = np.nansum(trans_wvs * photfilter_f(trans_wvs)) / np.nansum(photfilter_f(trans_wvs))
    grid_dwvs = grid_wvs[1::] - grid_wvs[0:np.size(grid_wvs) - 1]
    grid_dwvs = np.insert(grid_dwvs, 0, grid_dwvs[0])
    filter_norm = np.nansum((grid_dwvs * u.um) * photfilter_f(grid_wvs))
    Flambda = np.nansum((grid_dwvs * u.um)[None, None, :] * photfilter_f(grid_wvs)[None, None, :] * (
                grid_specs * u.W * u.m ** -2 / u.um), axis=2) / filter_norm
    Fnu = Flambda * (photfilter_wv0 * u.um) ** 2 / const.c  # from Flambda back to Fnu
    grid_specs = grid_specs / Fnu[:, :, None].to(u.MJy).value
    myinterpgrid2 = RegularGridInterpolator((grid_temps, grid_loggs), grid_specs, method="linear",
                                            bounds_error=False, fill_value=np.nan)
    # planet_f = interp1d(grid_wvs, myinterpgrid((1000, 5.0)), bounds_error=False, fill_value=0)

    fig3 = plt.figure(3, figsize=(6, 4.5))
    # exit()
    gs_snrmaps = gridspec.GridSpec(3,3, height_ratios=[0.05,1,1], width_ratios=[1,1,1])
    gs_snrmaps.update(left=0.1, right=0.95, bottom=0.125, top=0.90, wspace=0.0, hspace=0.0)
    fig1 = plt.figure(1, figsize=(6,4.5))
    fig11 = plt.figure(11, figsize=(6,4.5))
    fig2 = plt.figure(2, figsize=(6,4.5))
    fig12 = plt.figure(12, figsize=(6,4.5))
    contrast_5sig_1D_seps = np.arange(0,3,0.1)
    contrast_5sig_1D_med_array = np.zeros((np.size(contrast_5sig_1D_seps),np.size(teffs_list)))+np.nan
    contrast_5sig_1D_med_array_filters = {}
    for photfilter_name in ["F444W", "F356W", "Lp", "Mp"]:
        contrast_5sig_1D_med_array_filters[photfilter_name] = np.zeros((np.size(contrast_5sig_1D_seps), np.size(teffs_list))) + np.nan
    # for myteffid,myteff in enumerate([500,1000,1500,2000,2500,3000]):#
    for myteffid,myteff in enumerate(teffs_list):

        modelteff_NIRSpec_flux_MJy = {}
        # grid_dwvs = grid_wvs[1::] - grid_wvs[0:np.size(grid_wvs) - 1]
        # grid_dwvs = np.insert(grid_dwvs, 0, grid_dwvs[0])
        for photfilter_name in ["F360M", "F410M", "F430M", "F460M", "F444W", "F356W","Lp","Mp"]:
            if photfilter_name == "Lp":
                photfilter = os.path.join(external_dir,"Keck_NIRC2.Lp.dat")
            elif photfilter_name == "Mp":
                photfilter = os.path.join(external_dir,"Paranal_NACO.Mp.dat")
            else:
                photfilter = os.path.join(external_dir,"JWST_NIRCam." + photfilter_name + ".dat")
            filter_arr = np.loadtxt(photfilter)
            trans_wvs = filter_arr[:, 0] / 1e4
            trans = filter_arr[:, 1]
            photfilter_f = interp1d(trans_wvs, trans, bounds_error=False, fill_value=0)
            photfilter_wv0 = np.nansum(trans_wvs * photfilter_f(trans_wvs)) / np.nansum(photfilter_f(trans_wvs))
            bandpass = np.where(photfilter_f(trans_wvs) / np.nanmax(photfilter_f(trans_wvs)) > 0.01)
            photfilter_wvmin, photfilter_wvmax = trans_wvs[bandpass[0][0]], trans_wvs[bandpass[0][-1]]

            filter_norm = np.nansum((grid_dwvs * u.um) * photfilter_f(grid_wvs))
            if myteff<1600:
                model_spec = myinterpgrid((myteff, 5.0))
            else:
                model_spec = myinterpgrid2((myteff, 5.0))
            Flambda = np.nansum((grid_dwvs * u.um) * photfilter_f(grid_wvs) * (model_spec*u.W * (u.m ** -2) / u.um)) / filter_norm
            Fnu = Flambda * (photfilter_wv0 * u.um) ** 2 / const.c  # from Flambda back to Fnu
            Fnu_Mjy = Fnu.to(u.MJy).value

            modelteff_NIRSpec_flux_MJy[photfilter_name] = Fnu_Mjy
        print(myteff, modelteff_NIRSpec_flux_MJy)
        # continue
        # exit()

        plt.figure(10)

        F460M_to_F444W = modelteff_NIRSpec_flux_MJy["F444W"]/modelteff_NIRSpec_flux_MJy["F460M"]
        nrs2_flux_filename = os.path.join(out_png,"HD19467B_FM_F460M_nrs2"+"_"+"{0}K".format(myteff)+"_flux.fits")
        with fits.open(nrs2_flux_filename) as hdulist:
            nrs2_flux = hdulist[0].data*F460M_to_F444W
        nrs2_fluxerr_filename = os.path.join(out_png,"HD19467B_FM_F460M_nrs2"+"_"+"{0}K".format(myteff)+"_fluxerr.fits")
        with fits.open(nrs2_fluxerr_filename) as hdulist:
            nrs2_fluxerr = hdulist[0].data*F460M_to_F444W
            plt.subplot(2,2,1)
            plt.plot(hdulist[0].data[nrs2_fluxerr.shape[0]//2,:],label="nrs2 {0}".format(myteff))
            plt.yscale("log")
            plt.subplot(2,2,3)
            plt.plot(hdulist[0].data[:,nrs2_fluxerr.shape[1]//2],label="nrs2 {0}".format(myteff))
            plt.yscale("log")
            plt.legend()
        nrs2_RADec_filename = os.path.join(out_png,"HD19467B_FM_F460M_nrs2"+"_"+"{0}K".format(myteff)+"_RADec.fits")
        with fits.open(nrs2_RADec_filename) as hdulist:
            nrs2_RA_grid = hdulist[0].data
            nrs2_Dec_grid = hdulist[1].data
            nrs2_RA_vec =   nrs2_RA_grid[0,:]
            nrs2_Dec_vec =   nrs2_Dec_grid[:,0]
        dra = nrs2_RA_vec[1]-nrs2_RA_vec[0]
        ddec = nrs2_Dec_vec[1]-nrs2_Dec_vec[0]
        rs_grid = np.sqrt(nrs2_RA_grid**2+nrs2_Dec_grid**2)
        nrs2_hist_filename = os.path.join(out_png,"HD19467B_FM_F460M_nrs2"+"_"+"{0}K".format(myteff)+"_hist.fits")
        with fits.open(nrs2_hist_filename) as hdulist:
            nrs2_bins = hdulist[0].data
            nrs2_hist = hdulist[1].data

        F360M_to_F444W = modelteff_NIRSpec_flux_MJy["F444W"]/modelteff_NIRSpec_flux_MJy["F360M"]

        nrs1_flux_filename = os.path.join(out_png,"HD19467B_FM_F360M_nrs1"+"_"+"{0}K".format(myteff)+"_flux.fits")
        with fits.open(nrs1_flux_filename) as hdulist:
            nrs1_flux = hdulist[0].data*F360M_to_F444W
        nrs1_fluxerr_filename = os.path.join(out_png,"HD19467B_FM_F360M_nrs1"+"_"+"{0}K".format(myteff)+"_fluxerr.fits")
        with fits.open(nrs1_fluxerr_filename) as hdulist:
            nrs1_fluxerr = hdulist[0].data*F360M_to_F444W
            plt.subplot(2,2,2)
            plt.plot(hdulist[0].data[nrs1_fluxerr.shape[0]//2,:],label="nrs1 {0}".format(myteff))
            plt.yscale("log")
            plt.subplot(2,2,4)
            plt.plot(hdulist[0].data[:,nrs1_fluxerr.shape[1]//2],label="nrs1 {0}".format(myteff))
            plt.yscale("log")
            plt.legend()
        nrs1_RADec_filename = os.path.join(out_png,"HD19467B_FM_F360M_nrs1"+"_"+"{0}K".format(myteff)+"_RADec.fits")
        with fits.open(nrs1_RADec_filename) as hdulist:
            nrs1_RA_grid = hdulist[0].data
            nrs1_Dec_grid = hdulist[1].data
            nrs1_RA_vec =   nrs1_RA_grid[0,:]
            nrs1_Dec_vec =   nrs1_Dec_grid[:,0]
        nrs1_hist_filename = os.path.join(out_png,"HD19467B_FM_F360M_nrs1"+"_"+"{0}K".format(myteff)+"_hist.fits")
        with fits.open(nrs1_hist_filename) as hdulist:
            nrs1_bins = hdulist[0].data
            nrs1_hist = hdulist[1].data

        if 1:
            deno = (1/nrs1_fluxerr**2 + 1/nrs2_fluxerr**2)
            combined_flux = (nrs1_flux/nrs1_fluxerr**2 + nrs2_flux/nrs2_fluxerr**2)/deno
            combined_flux_err = 1/np.sqrt(deno)
            combined_SNR = combined_flux/combined_flux_err
        else:
            combined_flux = nrs2_flux
            combined_flux_err = nrs2_fluxerr
            combined_SNR = combined_flux/combined_flux_err


        contrast_5sig_combined  = 5*combined_flux_err/HD19467_flux_MJy["F444W"]
        contrast_5sig_combined[np.where(~np.isfinite(contrast_5sig_combined))] = np.nan
        contrast_5sig_1D_med = np.zeros(contrast_5sig_1D_seps.shape)+np.nan
        contrast_5sig_1D_min = np.zeros(contrast_5sig_1D_seps.shape)+np.nan
        for k,sep in enumerate(contrast_5sig_1D_seps):
            # mask=copy(contrast_5sig_combined)
            # mask[np.where((rs_grid>(sep-0.05))*(rs_grid<(sep+0.05)))] = np.nan
            # print( np.nanmedian(contrast_5sig_combined[np.where((rs_grid>(sep-0.05))*(rs_grid<(sep+0.05)))]))
            # plt.imshow(mask,origin="lower")
            # plt.show()
            contrast_5sig_1D_med[k] = np.nanmedian(contrast_5sig_combined[np.where((rs_grid>(sep-0.05))*(rs_grid<(sep+0.05)))])
            contrast_5sig_1D_min[k] = np.nanmin(contrast_5sig_combined[np.where((rs_grid>(sep-0.05))*(rs_grid<(sep+0.05)))])
        print(contrast_5sig_1D_med_array.shape)
        contrast_5sig_1D_med_array[:,myteffid] = contrast_5sig_1D_med
        fm_r2comp_grid = np.sqrt((nrs1_RA_grid-ra_offset)**2+(nrs1_Dec_grid-dec_offset)**2)
        snr_map_masked = copy(combined_SNR)
        nan_mask_boxsize = 5
        snr_map_masked[np.where(
            np.isnan(correlate2d(snr_map_masked, np.ones((nan_mask_boxsize, nan_mask_boxsize)), mode="same")))] = np.nan
        snr_map_masked[np.where((fm_r2comp_grid < 0.7))] = np.nan
        SNR_std = np.nanstd(snr_map_masked)
        print(myteff,np.nanstd(snr_map_masked))
        if 0:
            combined_flux_err = combined_flux_err*SNR_std
            contrast_5sig_combined = contrast_5sig_combined*SNR_std
            contrast_5sig_1D_med = contrast_5sig_1D_med*SNR_std
            contrast_5sig_1D_min = contrast_5sig_1D_min*SNR_std


        plt.figure(1, figsize=(6, 4.5))
        # plt.scatter(rs_grid,contrast_5sig_combined,s=0.2,c=color_list[myteffid],alpha=1)#,label="5$\sigma$ - Forward Model"
        wheregood = np.where(contrast_5sig_1D_seps>0.0)
        plt.plot(contrast_5sig_1D_seps[wheregood],contrast_5sig_1D_med[wheregood],label="{0}K".format(myteff),alpha=1,c=color_list[myteffid],linestyle=linelist_list[myteffid],linewidth=2)
        # plt.plot(contrast_5sig_1D_seps,contrast_5sig_1D_min,label="5$\sigma$ - {0}K - Best".format(myteff),alpha=0.5,c=color_list[myteffid],linestyle="--")

        # wheregood = np.where(contrast_5sig_1D_seps>0.5)
        # plt.plot(contrast_5sig_1D_seps[wheregood],contrast_5sig_1D_med[wheregood]/10.,alpha=0.5,c=color_list[myteffid],linestyle=linelist_list[myteffid],linewidth=1)
        # wheregood = np.where(contrast_5sig_1D_seps>0.0)
        # plt.plot(contrast_5sig_1D_seps[wheregood],contrast_5sig_1D_med[wheregood]*10.,alpha=0.5,c=color_list[myteffid],linestyle=linelist_list[myteffid],linewidth=1)

        for photfiltid, photfilter_name in enumerate(["F444W", "F356W", "Lp", "Mp"]):
            # contrast_5sig_1D_med_array_filters[photfilter_name] = np.zeros(
            #     (np.size(contrast_5sig_1D_seps), np.size(teffs_list))) + np.nan
            plt.figure(21+photfiltid, figsize=(6, 4.5))
            scaling = (modelteff_NIRSpec_flux_MJy[photfilter_name]/modelteff_NIRSpec_flux_MJy["F444W"])/(HD19467_flux_MJy[photfilter_name]/HD19467_flux_MJy["F444W"])
            contrast_5sig_1D_med_array_filters[photfilter_name][:,myteffid] = contrast_5sig_1D_med*scaling
            # plt.scatter(rs_grid,contrast_5sig_combined,s=0.2,c=color_list[myteffid],alpha=1)#,label="5$\sigma$ - Forward Model"
            wheregood = np.where(contrast_5sig_1D_seps>0.0)
            plt.plot(contrast_5sig_1D_seps[wheregood],contrast_5sig_1D_med[wheregood]*scaling,label="{0}K".format(myteff),alpha=1,c=color_list[myteffid],linestyle=linelist_list[myteffid],linewidth=2)

        plt.figure(12, figsize=(6, 4.5))
        hist, bins = np.histogram(snr_map_masked[np.where(np.isfinite(snr_map_masked))], bins=20 * 3, range=(-10, 10))
        bin_centers = (bins[1::] + bins[0:np.size(bins) - 1]) / 2.
        plt.plot(bin_centers, hist / (np.nansum(hist) * (bins[1] - bins[0])), label="{0}K".format(myteff), color=color_list[myteffid],
                 linestyle=linelist_list[myteffid])

        plt.figure(2, figsize=(6, 4.5))
        plt.subplot(1,2,1)
        hist, bins = np.histogram(snr_map_masked[np.where(np.isfinite(snr_map_masked))], bins=20 * 3, range=(-10, 10))
        bin_centers = (bins[1::] + bins[0:np.size(bins) - 1]) / 2.
        plt.plot(bin_centers, hist / (np.nansum(hist) * (bins[1] - bins[0])), label="{0}K".format(myteff), color=color_list[myteffid],
                 linestyle=linelist_list[myteffid])
        # plt.plot(nrs1_bins, nrs1_hist, label="NRS1", color=color_list[1], linestyle=":")
        # plt.plot(nrs2_bins, nrs2_hist, label="NRS2", color="pink", linestyle="-.")
        plt.subplot(1,2,2)
        hist, bins = np.histogram(snr_map_masked[np.where(np.isfinite(snr_map_masked))]/SNR_std, bins=20 * 3, range=(-10, 10))
        bin_centers = (bins[1::] + bins[0:np.size(bins) - 1]) / 2.
        plt.plot(bin_centers, hist / (np.nansum(hist) * (bins[1] - bins[0])), label="{0}K".format(myteff), color=color_list[myteffid],
                 linestyle=linelist_list[myteffid])

        print(1+int(myteffid//3), (myteffid%3))
        plt.figure(3)
        ax1 = plt.subplot(gs_snrmaps[1+int(myteffid//3), (myteffid%3)])
        # plt.subplot(2,3,myteffid+1)
        # plt.title("{0}K".format(myteff))
        plt3 = plt.imshow(combined_SNR,origin="lower",extent=[nrs2_RA_vec[0]-dra/2.,nrs2_RA_vec[-1]+dra/2.,nrs2_Dec_vec[0]-ddec/2.,nrs2_Dec_vec[-1]+ddec/2.])
        plt.clim([-2,5])
        plt.gca().set_aspect('equal')
        plt.xlim([-2.5,2.5])
        plt.ylim([-3.0,2.])
        plt.gca().invert_xaxis()
        plt.text(0.02, 0.98, r"{0}K".format(myteff), fontsize=fontsize, ha='left', va='top', transform=plt.gca().transAxes,color="Black")
        if myteffid >=3:
            plt.xticks([-2,-1,0,1,2])
            plt.xlabel("$\Delta$RA (as)",fontsize=fontsize)
            plt.gca().tick_params(axis='x', labelsize=fontsize)
        else:
            plt.xticks([])
        if myteffid ==0:
            plt.yticks([-3,-2,-1,0,1,2])
            plt.ylabel("$\Delta$Dec (as)",fontsize=fontsize)
            plt.gca().tick_params(axis='y', labelsize=fontsize)
        elif myteffid ==3:
            plt.yticks([-3,-2,-1,0,1])
            plt.ylabel("$\Delta$Dec (as)",fontsize=fontsize)
            plt.gca().tick_params(axis='y', labelsize=fontsize)
        else:
            plt.yticks([])

        if myteffid <3:
            cbax = plt.subplot(gs_snrmaps[0, myteffid])
            cb = Colorbar(ax=cbax, mappable=plt3, orientation='horizontal', ticklocation='top')
            cb.set_label(r'S/N', labelpad=5, fontsize=fontsize)
            # cb.set_ticks([-5,-2.5,0,2.5])  # xticks[1::]
            if myteffid == 0:
                cb.set_ticks([-2,-1,0,1,2,3,4,5])  # xticks[1::]
            else:
                cb.set_ticks([-1,0,1,2,3,4,5])  # xticks[1::]
            plt.gca().tick_params(axis='x', labelsize=fontsize)
            plt.gca().tick_params(axis='y', labelsize=fontsize)
    # plt.text(3, 2e-5, "K=10.4\nExtrapolated", fontsize=fontsize, ha='left', va='top',color="Black")
    # plt.text(3, 2e-6, "K=5.4\nThis work", fontsize=fontsize, ha='left', va='top',color="Black")
    # plt.text(3, 2e-7, "K=0.4\nExtrapolated", fontsize=fontsize, ha='left', va='top',color="Black")

    import csv

    with open(os.path.join(out_png, "HD19467B_5sig_contrast_vs_Teff.csv"), 'w', newline='') as file:
        writer = csv.writer(file,delimiter='\t')
        writer.writerow(["# Ruffio et al. 2023 (in prep.)"])
        writer.writerow(["# Detection sensitivity of NIRSpec as a function of effective temperature of the companion model (BTSettl - Allard+2003)"])
        writer.writerow(["# GTO 1414 - HD 19467 - Kmag=5.4 - Tint=35 min (no overheads included)"])
        writer.writerow(["# (1st column) Separation to the star in arcsecond"])
        writer.writerow(["# (other columns) 5sigma detection limits of NIRSpec IFU expressed as a companion-to-star flux ratio in F444W"])
        writer.writerow(["Sep"]+["{0}K".format(teff) for teff in teffs_list])
        writer.writerows(np.concatenate([contrast_5sig_1D_seps[1::,None],contrast_5sig_1D_med_array[1::,:]],axis=1))

    # np.set_printoptions(precision=4)
    with open(os.path.join(out_png, "HD19467B_5sig_contrast_vs_Teff_allfilters.csv"), 'w', newline='') as file:
        writer = csv.writer(file,delimiter='\t')
        writer.writerow(["# Ruffio et al. 2023 (in prep.)"])
        writer.writerow(["# Detection sensitivity of NIRSpec as a function of effective temperature of the companion model (BTSettl - Allard+2003)"])
        writer.writerow(["# GTO 1414 - HD 19467 - Kmag=5.4 - Tint=35 min (no overheads included)"])
        writer.writerow(["# (1st column) Separation to the star in milli-arcsecond"])
        writer.writerow(["# (other columns) 5sigma detection limits of NIRSpec IFU expressed as a companion-to-star flux ratio in the indicated filter"])
        writer.writerow(["# Filter reference: http://svo2.cab.inta-csic.es/theory/fps/"])
        writer.writerow(["# F356W = JWST_NIRCam.F356W.dat"])
        writer.writerow(["# F444W = JWST_NIRCam.F444W.dat"])
        writer.writerow(["# Lp = Keck_NIRC2.Lp.dat "])
        writer.writerow(["# Mp = Paranal_NACO.Mp.dat"])
        writer.writerow(["Sep_mas"]+["{0} - Teff = {1}K".format(photfilter_name,teff) for photfilter_name in ["F356W", "F444W", "Lp", "Mp"] for teff in teffs_list ])
        outarray = np.round(contrast_5sig_1D_seps[1::,None]*1000).astype(int)
        for photfiltid, photfilter_name in enumerate(["F356W","F444W", "Lp", "Mp"]):
            outarray = np.concatenate([outarray, contrast_5sig_1D_med_array_filters[photfilter_name][1::, :]], axis=1)
        # outarray = np.around(outarray, decimals=2, out=None)
        # outarray = ((np.round(outarray*1e8).astype(int)).astype(float))*1e-8
        writer.writerows(outarray)


    plt.figure(3)
    out_filename = os.path.join(out_png, "snrmap_vs_teff.png")
    print("Saving " + out_filename)
    plt.savefig(out_filename, dpi=300)
    plt.savefig(out_filename.replace(".png", ".pdf"))

    plt.figure(12)
    plt.plot(bin_centers, 1 / np.sqrt(2 * np.pi) * np.exp(-0.5 * (bin_centers - 0.0) ** 2), color="black",
             linestyle="--", label="Gaussian")
    plt.yscale("log")
    plt.xlim([-5, 5])
    plt.ylim([1e-4, 1])
    plt.xlabel("SNR", fontsize=fontsize)
    plt.ylabel("PDF", fontsize=fontsize)
    plt.legend(handlelength=3)
    plt.gca().tick_params(axis='x', labelsize=fontsize)
    plt.gca().tick_params(axis='y', labelsize=fontsize)

    out_filename = os.path.join(out_png, "hist_vs_teff.png")
    print("Saving " + out_filename)
    plt.savefig(out_filename, dpi=300)
    plt.savefig(out_filename.replace(".png", ".pdf"))

    plt.figure(2)
    plt.subplot(1,2,1)
    plt.text(0.02, 0.98, r"Before S/N normalization", fontsize=fontsize, ha='left', va='top', transform=plt.gca().transAxes,color="Black")
    plt.subplot(1,2,2)
    plt.text(0.02, 0.98, r"After S/N normalization", fontsize=fontsize, ha='left', va='top', transform=plt.gca().transAxes,color="Black")

    plt.subplot(1,2,1)
    plt.plot(bin_centers, 1 / np.sqrt(2 * np.pi) * np.exp(-0.5 * (bin_centers - 0.0) ** 2), color="black",
             linestyle="--", label="Gaussian")
    plt.yscale("log")
    plt.xlim([-5, 5])
    plt.ylim([1e-4, 1])
    plt.xlabel("SNR", fontsize=fontsize)
    plt.ylabel("PDF", fontsize=fontsize)
    plt.legend(handlelength=3)
    plt.gca().tick_params(axis='x', labelsize=fontsize)
    plt.gca().tick_params(axis='y', labelsize=fontsize)
    plt.subplot(1,2,2)
    plt.plot(bin_centers, 1 / np.sqrt(2 * np.pi) * np.exp(-0.5 * (bin_centers - 0.0) ** 2), color="black",
             linestyle="--", label="Gaussian")
    plt.yscale("log")
    plt.xlim([-5, 5])
    plt.ylim([1e-4, 1])
    plt.yticks([])
    plt.xticks([-2.5,0,2.5,5])
    plt.xlabel("SNR", fontsize=fontsize)
    # plt.ylabel("PDF", fontsize=fontsize)
    plt.legend(handlelength=3)
    plt.gca().tick_params(axis='x', labelsize=fontsize)
    # plt.gca().tick_params(axis='y', labelsize=fontsize)
    plt.subplots_adjust(wspace=0, hspace=0)
    # plt.tight_layout()

    out_filename = os.path.join(out_png, "hist_vs_teff_norma.png")
    print("Saving " + out_filename)
    plt.savefig(out_filename, dpi=300)
    plt.savefig(out_filename.replace(".png", ".pdf"))

    plt.figure(1)

    F444W_Carter = np.loadtxt("/stow/jruffio/data/JWST/nirspec/HD_19467/breads/figures/HIP65426_ERS_CONTRASTS/F444W_ADI+RDI.txt")
    F444W_Carter_seps = F444W_Carter[:, 0]
    F444W_Carter_cons = F444W_Carter[:, 1]*np.sqrt(1236/2100*10**((5.4-6.771)/2.5))

    plt.plot(F444W_Carter_seps,F444W_Carter_cons,alpha=1,c="black",linestyle="-.",label= 'NIRCam')#,label="NIRCam ADI+RDI"
    # plt.text(1.2,4e-6, 'NIRCam Carter+2023', fontsize=fontsize, ha='left', va='center',color="black",rotation=-11)# \n 36-166 $\mu$Jy
    plt.fill_between([0,0.3],[10**(-10),10**(-10)],[1,1],color="grey",alpha=0.2)

    plt.yscale("log")
    plt.xscale("log")
    plt.xlim([0.1,3])
    # plt.xlim([0.1,10])
    plt.ylim([0.6*10**(-6),10**(-3)])
    plt.text(0.02, 0.02, r"HD 19467 - Kmag=5.4 - $T_{\mathrm{int}}=35$ min", fontsize=fontsize, ha='left', va='bottom', transform=plt.gca().transAxes,color="Black")
    plt.xlabel("Separation (as)",fontsize=fontsize)
    plt.ylabel("5$\sigma$ Flux ratio ({0})".format("F444W"),fontsize=fontsize)
    plt.legend(loc="upper right",frameon=True,fontsize=10,handlelength=4)
    plt.gca().tick_params(axis='x', labelsize=fontsize)
    plt.gca().tick_params(axis='y', labelsize=fontsize)

    out_filename = os.path.join(out_png, "contrast_vs_teff.png")
    print("Saving " + out_filename)
    plt.savefig(out_filename, dpi=300)
    plt.savefig(out_filename.replace(".png", ".pdf"))


    for photfiltid, photfilter_name in enumerate(["F444W", "F356W", "Lp", "Mp"]):
        # contrast_5sig_1D_med_array_filters[photfilter_name] = np.zeros(
        #     (np.size(contrast_5sig_1D_seps), np.size(teffs_list))) + np.nan
        plt.figure(21+photfiltid)
        if photfilter_name =="F444W":
            plt.plot(F444W_Carter_seps,F444W_Carter_cons,alpha=1,c="black",linestyle="-.",label= 'NIRCam')#,label="NIRCam ADI+RDI"
        plt.yscale("log")
        plt.xscale("log")
        plt.xlim([0.1,3])
        # plt.xlim([0.1,10])
        plt.ylim([1*10**(-7),10**(-3)])
        plt.text(0.02, 0.02, r"HD 19467 - Kmag=5.4 - $T_{\mathrm{int}}=35$ min", fontsize=fontsize, ha='left', va='bottom', transform=plt.gca().transAxes,color="Black")
        plt.fill_between([0,0.3],[10**(-10),10**(-10)],[1,1],color="grey",alpha=0.2)
        plt.xlabel("Separation (as)",fontsize=fontsize)
        plt.ylabel("5$\sigma$ Flux ratio ({0})".format(photfilter_name),fontsize=fontsize)
        plt.legend(loc="upper right",frameon=True,fontsize=10,handlelength=4)
        plt.gca().tick_params(axis='x', labelsize=fontsize)
        plt.gca().tick_params(axis='y', labelsize=fontsize)

        out_filename = os.path.join(out_png, "contrast_vs_teff_"+photfilter_name+".png")
        print("Saving " + out_filename)
        plt.savefig(out_filename, dpi=300)
        plt.savefig(out_filename.replace(".png", ".pdf"))


    # minwv, maxwv = np.min(dataobj.wavelengths), np.max(dataobj.wavelengths)
    # /stow/jruffio/data/JWST/nirspec/HD_19467/breads/external/BT-Settl15_0.5-6.0um_Teff500_1600_logg3.5_5.0.hdf5
    # with h5py.File(os.path.join(utils_dir,"BT-Settl_M-0.0_a+0.0_3-6um_500-2500K.hdf5"), 'r') as hf:
    # with h5py.File(os.path.join(external_dir,"BT-Settl15_0.5-6.0um_Teff500_1600_logg3.5_5.0_NIRSpec.hdf5"), 'r') as hf:
    # with h5py.File(os.path.join(external_dir,"BT-Settl_3-6um_Teff500_1600_logg3.5_5.0_NIRSpec_3-6um.hdf5"), 'r') as hf:

    photfilter_name= "F444W"
    photfilter = "/stow/jruffio/data/JWST/nirspec/HD_19467/breads/external/JWST_NIRCam." + photfilter_name + ".dat"
    filter_arr = np.loadtxt(photfilter)
    trans_wvs = filter_arr[:, 0] / 1e4
    trans = filter_arr[:, 1]
    photfilter_f = interp1d(trans_wvs, trans, bounds_error=False, fill_value=0)
    photfilter_wv0 = np.nansum(trans_wvs * photfilter_f(trans_wvs)) / np.nansum(photfilter_f(trans_wvs))
    bandpass = np.where(photfilter_f(trans_wvs) / np.nanmax(photfilter_f(trans_wvs)) > 0.01)
    photfilter_wvmin, photfilter_wvmax = trans_wvs[bandpass[0][0]], trans_wvs[bandpass[0][-1]]

    grid_filename2 = os.path.join(external_dir, "BT-Settlchris_0.5-6.0um_Teff1500_3000_logg3.5_5.5_NIRSpec.hdf5")
    with h5py.File(grid_filename2, 'r') as hf:
        grid_specs = np.array(hf.get("spec"))
        grid_temps = np.array(hf.get("temps"))
        grid_loggs = np.array(hf.get("loggs"))
        grid_wvs2 = np.array(hf.get("wvs"))
    grid_dwvs = grid_wvs2[1::] - grid_wvs2[0:np.size(grid_wvs2) - 1]
    grid_dwvs = np.insert(grid_dwvs, 0, grid_dwvs[0])
    filter_norm = np.nansum((grid_dwvs * u.um) * photfilter_f(grid_wvs2))
    Flambda = np.nansum((grid_dwvs * u.um)[None, None, :] * photfilter_f(grid_wvs2)[None, None, :] * (
                grid_specs * u.W * u.m ** -2 / u.um), axis=2) / filter_norm
    Fnu = Flambda * (photfilter_wv0 * u.um) ** 2 / const.c  # from Flambda back to Fnu
    grid_specs = 1e6*grid_specs / Fnu[:, :, None].to(u.MJy).value
    # grid_specs = grid_specs / np.polyval([-0.03128495, 1.09007397], grid_wvs2)[None, None, :]
    myinterpgrid2 = RegularGridInterpolator((grid_temps, grid_loggs), grid_specs, method="linear", bounds_error=False,
                                           fill_value=np.nan)

    grid_filename1 = os.path.join(external_dir, "BT-Settlchris_0.5-6.0um_Teff500_1600_logg3.5_5.0_NIRSpec.hdf5")
    with h5py.File(grid_filename1, 'r') as hf:
        grid_specs = np.array(hf.get("spec"))
        grid_temps = np.array(hf.get("temps"))
        grid_loggs = np.array(hf.get("loggs"))
        grid_wvs1 = np.array(hf.get("wvs"))
    grid_dwvs = grid_wvs1[1::] - grid_wvs1[0:np.size(grid_wvs1) - 1]
    grid_dwvs = np.insert(grid_dwvs, 0, grid_dwvs[0])
    filter_norm = np.nansum((grid_dwvs * u.um) * photfilter_f(grid_wvs1))
    Flambda = np.nansum((grid_dwvs * u.um)[None, None, :] * photfilter_f(grid_wvs1)[None, None, :] * (
                grid_specs * u.W * u.m ** -2 / u.um), axis=2) / filter_norm
    Fnu = Flambda * (photfilter_wv0 * u.um) ** 2 / const.c  # from Flambda back to Fnu
    grid_specs = 1e6*grid_specs / Fnu[:, :, None].to(u.MJy).value
    # grid_specs = grid_specs / np.polyval([-0.03128495, 1.09007397], grid_wvs1)[None, None, :]
    myinterpgrid1 = RegularGridInterpolator((grid_temps, grid_loggs), grid_specs, method="linear", bounds_error=False,
                                           fill_value=np.nan)

    teff, logg, vsini, rv, dra_comp, ddec_comp = myteff, 5.0, 0.0, None, None, None
    fix_parameters = [teff, logg, vsini, rv, dra_comp, ddec_comp]

    # print(grid_temps)
    # print(grid_loggs)
    # wherewvs = np.where((grid_wvs>3.8)*(grid_wvs<5.2))
    from breads.utils import broaden

    plt.figure(4, figsize=(6, 4.5))
    for myteffid,myteff in enumerate([500,1000,1500,2000,2500,3000]):#
        if myteff <1500:
            planet_model = myinterpgrid1([myteff,5.0])[0]
            plt.plot(grid_wvs1,planet_model,label="{0}K".format(myteff),c=color_list[myteffid],linestyle=linelist_list[myteffid],alpha=1,linewidth=1)
        else:
            planet_model = myinterpgrid2([myteff,5.0])[0]
            plt.plot(grid_wvs2,planet_model,label="{0}K".format(myteff),c=color_list[myteffid],linestyle=linelist_list[myteffid],alpha=1,linewidth=1)
        # planet_model_broaden = broaden(grid_wvs, planet_model, R=150, mppool=None, kernel=None)
        # print(np.sqrt(np.nansum((planet_model[wherewvs]-planet_model_broaden[wherewvs]) ** 2)),np.sqrt(np.nansum(planet_model[wherewvs] ** 2)))
    plt.legend(loc="upper right",frameon=True,fontsize=10,handlelength=4)
    plt.ylabel("Flux (W/m$^2$/$\mu$m)",fontsize=fontsize)
    plt.xlabel("Wavelength ($\mu$m)",fontsize=fontsize)
    plt.gca().tick_params(axis='x', labelsize=fontsize)
    plt.gca().tick_params(axis='y', labelsize=fontsize)
    plt.xlim([2.8,5.3])
    plt.ylim([0.0,0.45])

    out_filename = os.path.join(out_png, "BTSettl_vs_teff.png")
    print("Saving " + out_filename)
    plt.savefig(out_filename, dpi=300)
    plt.savefig(out_filename.replace(".png", ".pdf"))

    plt.show()
    exit()

