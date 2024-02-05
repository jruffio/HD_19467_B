import numpy as np
import matplotlib.pyplot as plt
from  scipy.interpolate import interp1d
import os
import astropy.io.fits as fits
from glob import glob
import h5py
from scipy.signal import correlate2d

from copy import copy
import matplotlib
matplotlib.use('Qt5Agg')

if __name__ == "__main__":
    try:
        import mkl
        mkl.set_num_threads(1)
    except:
        pass
    os.environ["WEBBPSF_PATH"] = "/stow/jruffio/data/webbPSF/webbpsf-data"

    numthreads = 20

    # RA Dec offset of the companion
    ra_offset = -1.38  # ra offset in as
    dec_offset = -0.92  # dec offset in as
    # Reference: Brandt et al. 2021


    external_dir = "/stow/jruffio/data/JWST/nirspec/HD_19467/breads/external/"
    out_dir = "/stow/jruffio/data/JWST/nirspec/HD_19467/breads/20240127_out/xy/"
    out_png = out_dir.replace("xy","figures")
    if not os.path.exists(out_png):
        os.makedirs(out_png)


    ####################
    ## To be modified
    ####################
    # for plotting
    fontsize = 12
    # Directories to update
    os.environ["WEBBPSF_PATH"] = "/stow/jruffio/data/webbPSF/webbpsf-data"
    crds_dir="/stow/jruffio/data/JWST/crds_cache/"
    # External_dir should external files like the NIRCam filters
    external_dir = "/stow/jruffio/data/JWST/nirspec/HD_19467/breads/external/"
    # Science data: List of stage 2 cal.fits files
    filelist = glob("/stow/jruffio/data/JWST/nirspec/HD_19467/HD19467_onaxis_roll2/20240124_stage2_clean/jw01414004001_02101_*_nrs*_cal.fits")
    filelist.sort()
    for filename in filelist:
        print(filename)
    print("N files: {0}".format(len(filelist)))
    out_dir = "/stow/jruffio/data/JWST/nirspec/HD_19467/breads/20240201_out_fm/xy/"
    out_png = out_dir.replace("xy","figures")
    if not os.path.exists(out_png):
        os.makedirs(out_png)
    # Path to a s3d cube to extract an empirical PSF profile
    A0_filename = "/stow/jruffio/data/JWST/nirspec/A0_TYC 4433-1800-1/MAST_2023-04-23T0044/JWST/jw01128-o009_t007_nirspec_g395h-f290lp/jw01128-o009_t007_nirspec_g395h-f290lp_s3d.fits"
    ####################
    # detector,photfilter_name = "nrs1", "F360M"
    detector, photfilter_name = "nrs2", "F460M"
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
    HD19467_flux_MJy = {"F250M":3.51e-6, # in MJy, Ref Greenbaum+2023
                         "F300M":2.63e-6,
                         "F335M":2.10e-6,
                         "F360M":1.82e-6,
                         "F410M":1.49e-6,
                         "F430M":1.36e-6,
                         "F460M":1.12e-6}

    # photfilter = os.path.join(external_dir,"JWST_NIRCam."+photfilter_name+".dat")
    # filter_arr = np.loadtxt(photfilter)
    # trans_wvs = filter_arr[:,0]/1e4
    # trans = filter_arr[:,1]
    # photfilter_f = interp1d(trans_wvs,trans,bounds_error=False,fill_value=0)
    # photfilter_wv0 = np.nansum(trans_wvs*photfilter_f(trans_wvs))/np.nansum(photfilter_f(trans_wvs))
    # bandpass = np.where(photfilter_f(trans_wvs)/np.nanmax(photfilter_f(trans_wvs))>0.01)
    # photfilter_wvmin,photfilter_wvmax = trans_wvs[bandpass[0][0]],trans_wvs[bandpass[0][-1]]
    # print(photfilter_wvmin,photfilter_wvmax)


    hdulist_sc = fits.open(A0_filename)
    cube = hdulist_sc["SCI"].data
    im_psfprofile = np.nanmedian(cube,axis=0)
    im_psfprofile /= np.nanmax(im_psfprofile)
    kc,lc = np.unravel_index(np.nanargmax(im_psfprofile),im_psfprofile.shape)
    kvec = (np.arange(im_psfprofile.shape[0])-kc)*0.1
    lvec = (np.arange(im_psfprofile.shape[1])-lc)*0.1
    lgrid,kgrid = np.meshgrid(lvec,kvec)
    rgrid_psfprofile = np.sqrt(lgrid**2+kgrid**2)
    im_psfprofile[np.where(rgrid_psfprofile>1.5)] = np.nan

    outoftheoven_filelist = []
    for filename in filelist:
        if detector not in filename:
            continue
        print(os.path.join(out_dir,"*_"+os.path.basename(filename).replace(".fits","_out.fits")))
        tmp_filelist = glob(os.path.join(out_dir,"*_"+os.path.basename(filename).replace(".fits","_out.fits")))
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

        k,l,m = np.unravel_index(np.nanargmax(log_prob-log_prob_H0),log_prob.shape)
        print("best fit parameters: rv={0},y={1},x={2}".format(rvs[k],ras[l],decs[m]) )
        # print(np.nanmax(log_prob-log_prob_H0))
        # best_log_prob,best_log_prob_H0,_,_,_ = grid_search([[rvs[k]], [ras[l]], [decs[m]]], dataobj, fm_func, fm_paras, numthreads=None)
        # print(best_log_prob-best_log_prob_H0)

        linparas = np.swapaxes(linparas, 1, 2)
        linparas_err = np.swapaxes(linparas_err, 1, 2)
        log_prob = np.swapaxes(log_prob, 1, 2)
        log_prob_H0 = np.swapaxes(log_prob_H0, 1, 2)
        rchi2 = np.swapaxes(rchi2, 1, 2)

        dra=ras[1]-ras[0]
        ddec=decs[1]-decs[0]
        ras_grid,decs_grid = np.meshgrid(ras,decs)
        rs_grid = np.sqrt(ras_grid**2+decs_grid**2)

        plt.figure(1,figsize=(4*int(np.ceil(N_files/3.)),3*3))
        plt.subplot(3,int(np.ceil(N_files/3.)),fid+1)
        plt.title("{0} File {1}".format(detector,fid),fontsize=fontsize)
        fluxmap_list.append(linparas[k,:,:,0])
        fluxerrmap_list.append(linparas_err[k,:,:,0])
        snr_map = linparas[k,:,:,0]/linparas_err[k,:,:,0]
        plt.imshow(snr_map,origin="lower",extent=[ras[0]-dra/2.,ras[-1]+dra/2.,decs[0]-ddec/2.,decs[-1]+ddec/2.])
        plt.clim([-2,5])
        cbar = plt.colorbar()
        cbar.set_label("SNR",fontsize=fontsize)
        # plt.plot(out[:,0,0,2])
        plt.xlabel("dRA (as)",fontsize=fontsize)
        plt.ylabel("ddec (as)",fontsize=fontsize)
        plt.gca().tick_params(axis='x', labelsize=fontsize)
        plt.gca().tick_params(axis='y', labelsize=fontsize)

        plt.figure(2,figsize=(4*int(np.ceil(N_files/3.)),3*3))
        plt.subplot(3,int(np.ceil(N_files/3.)),fid+1)
        snr_map_masked = copy(snr_map)
        rs_comp_grid = np.sqrt((ras_grid-ra_offset)**2+(decs_grid-dec_offset)**2)
        nan_mask_boxsize=5
        snr_map_masked[np.where(np.isnan(correlate2d(snr_map_masked,np.ones((nan_mask_boxsize,nan_mask_boxsize)),mode="same")))] = np.nan
        snr_map_masked[np.where((rs_comp_grid < 0.7))] = np.nan
        snr_vals.append(snr_map_masked[np.where(np.isfinite(snr_map_masked))])
        hist, bins = np.histogram(snr_map_masked[np.where(np.isfinite(snr_map_masked))], bins=20 * 3,range=(-10, 10))
        bin_centers = (bins[1::] + bins[0:np.size(bins) - 1]) / 2.
        plt.plot(bin_centers, hist / (np.nansum(hist) * (bins[1] - bins[0])), label="snr map")
        plt.plot(bin_centers, 1 / np.sqrt(2 * np.pi) * np.exp(-0.5 * (bin_centers - 0.0) ** 2), color="black",
                 linestyle="--", label="Gaussian")
        plt.yscale("log")
        plt.xlim([-6, 6])
        plt.ylim([1e-4, 1])
        plt.xlabel("SNR",fontsize=fontsize)
        plt.ylabel("PDF",fontsize=fontsize)
        plt.legend()
        plt.gca().tick_params(axis='x', labelsize=fontsize)
        plt.gca().tick_params(axis='y', labelsize=fontsize)

        contrast_5sig  = 5*linparas_err[k,:,:,0]/HD19467_flux_MJy[photfilter_name]
        nan_mask_boxsize=2
        contrast_5sig[np.where(np.isnan(correlate2d(contrast_5sig,np.ones((nan_mask_boxsize,nan_mask_boxsize)),mode="same")))] = np.nan


        plt.figure(3,figsize=(4*int(np.ceil(N_files/3.)),3*3))
        plt.subplot(3,int(np.ceil(N_files/3.)),fid+1)
        plt.title("Flux map {0} File {1}".format(detector,fid),fontsize=fontsize)
        plt.imshow(linparas[k,:,:,0],origin="lower",extent=[ras[0]-dra/2.,ras[-1]+dra/2.,decs[0]-ddec/2.,decs[-1]+ddec/2.])
        if detector == "nrs1":
            plt.clim([-0.25e-11, 1e-11])
        elif detector == "nrs2":
            plt.clim([-1e-11,4e-11])
        cbar = plt.colorbar()
        cbar.set_label("Planet flux MJy ({0})".format(photfilter_name),fontsize=fontsize)
        plt.xlabel("dRA (as)",fontsize=fontsize)
        plt.ylabel("ddec (as)",fontsize=fontsize)
        plt.gca().tick_params(axis='x', labelsize=fontsize)
        plt.gca().tick_params(axis='y', labelsize=fontsize)


        plt.figure(4,figsize=(4*int(np.ceil(N_files/3.)),3*3))
        plt.subplot(3,int(np.ceil(N_files/3.)),fid+1)
        plt.title("Sensitivity {0} File {1}".format(detector,fid),fontsize=fontsize)
        plt.imshow(np.log10(contrast_5sig),origin="lower",extent=[ras[0]-dra/2.,ras[-1]+dra/2.,decs[0]-ddec/2.,decs[-1]+ddec/2.])
        if detector == "nrs1":
            plt.clim([-6.5,-5])
        elif detector == "nrs2":
            plt.clim([-6,-4.5])
        plt.xlabel("dRA (as)",fontsize=fontsize)
        plt.ylabel("ddec (as)",fontsize=fontsize)
        cbar = plt.colorbar()
        cbar.set_label("5-$\sigma$ Flux ratio log10 ({0})".format(photfilter_name),fontsize=fontsize)
        plt.gca().tick_params(axis='x', labelsize=fontsize)
        plt.gca().tick_params(axis='y', labelsize=fontsize)

        plt.figure(5,figsize=(5,5))
        plt.scatter(rs_grid,contrast_5sig,alpha=1,s=0.2,label="file {0}".format(fid))
        plt.title("5-$\sigma$ Sensitivity 1D {0}".format(detector),fontsize=fontsize)
        plt.yscale("log")
        if detector == "nrs1":
            plt.ylim([10**(-6.5),10**(-2.5)])
        elif detector == "nrs2":
            plt.ylim([10**(-6),10**(-2)])
        plt.xlabel("Separation (as)",fontsize=fontsize)
        plt.ylabel("5-$\sigma$ Flux ratio ({0})".format(photfilter_name),fontsize=fontsize)
        # plt.show()

    import datetime
    now = datetime.datetime.now()
    formatted_datetime = now.strftime("%Y%m%d_%H%M%S")

    plt.figure(1)
    plt.tight_layout()
    out_filename = os.path.join(out_png,formatted_datetime+"_"+detector+"_SNR.png")
    print("Saving " + out_filename)
    plt.savefig(out_filename)
    plt.savefig(out_filename.replace(".png",".pdf"))

    plt.figure(2)
    plt.tight_layout()
    out_filename = os.path.join(out_png,formatted_datetime+"_"+detector+"_hist.png")
    print("Saving " + out_filename)
    plt.savefig(out_filename)
    plt.savefig(out_filename.replace(".png",".pdf"))


    plt.figure(3)
    plt.tight_layout()
    out_filename = os.path.join(out_png,formatted_datetime+"_"+detector+"_fluxmap.png")
    print("Saving " + out_filename)
    plt.savefig(out_filename)
    plt.savefig(out_filename.replace(".png",".pdf"))

    plt.figure(4)
    plt.tight_layout()
    out_filename = os.path.join(out_png,formatted_datetime+"_"+detector+"_fluxerrmap.png")
    print("Saving " + out_filename)
    plt.savefig(out_filename)
    plt.savefig(out_filename.replace(".png",".pdf"))

    plt.figure(5)
    plt.scatter(rgrid_psfprofile.ravel(),im_psfprofile.ravel(),c="black",s=0.5,label="PSF profile")
    plt.legend(loc="upper left",frameon=True,fontsize=fontsize)
    plt.gca().tick_params(axis='x', labelsize=fontsize)
    plt.gca().tick_params(axis='y', labelsize=fontsize)
    plt.tight_layout()
    out_filename = os.path.join(out_png,formatted_datetime+"_"+detector+"_1d_sensitivity.png")
    print("Saving " + out_filename)
    plt.savefig(out_filename)
    plt.savefig(out_filename.replace(".png",".pdf"))

    fluxmap_arr = np.array(fluxmap_list)
    fluxerrmap_arr = np.array(fluxerrmap_list)
    fluxmap_combined = np.nansum(fluxmap_arr/fluxerrmap_arr**2,axis=0)/np.nansum(1/fluxerrmap_arr**2,axis=0)
    fluxerrmap_combined = 1/np.sqrt(np.nansum(1/fluxerrmap_arr**2,axis=0))
    snr_map_combined = fluxmap_combined/fluxerrmap_combined
    contrast_5sig_combined  = 5*fluxerrmap_combined/HD19467_flux_MJy[photfilter_name]

    plt.figure(6,figsize=(12,4))


    plt.subplot(1,3,1)
    plt.title("Combined Flux {0} {1}".format(detector,photfilter_name),fontsize=fontsize)
    plt.imshow(fluxmap_combined,origin="lower",extent=[ras[0]-dra/2.,ras[-1]+dra/2.,decs[0]-ddec/2.,decs[-1]+ddec/2.])
    plt.gca().invert_xaxis()
    if detector == "nrs1":
        plt.clim([-0.25e-11, 1e-11])
    elif detector == "nrs2":
        plt.clim([-1e-11,4e-11])
    cbar = plt.colorbar()
    cbar.set_label("Planet flux MJy ({0})".format(photfilter_name),fontsize=fontsize)
    plt.xlabel("dRA (as)",fontsize=fontsize)
    plt.ylabel("ddec (as)",fontsize=fontsize)

    plt.subplot(1,3,2)
    plt.title("Sensitivity {0} {1}".format(detector,photfilter_name),fontsize=fontsize)
    plt.imshow(np.log10(contrast_5sig_combined),origin="lower",extent=[ras[0]-dra/2.,ras[-1]+dra/2.,decs[0]-ddec/2.,decs[-1]+ddec/2.])
    plt.gca().invert_xaxis()
    if detector == "nrs1":
        plt.clim([-6.,-4.5])
    elif detector == "nrs2":
        plt.clim([-6.,-4.5])
    plt.xlabel("dRA (as)",fontsize=fontsize)
    plt.ylabel("ddec (as)",fontsize=fontsize)
    cbar = plt.colorbar()
    cbar.set_label("5-$\sigma$ Flux ratio log10 ({0})".format(photfilter_name),fontsize=fontsize)

    plt.subplot(1,3,3)
    plt.title("Combined SNR {0}".format(detector))
    plt.imshow(snr_map_combined,origin="lower",extent=[ras[0]-dra/2.,ras[-1]+dra/2.,decs[0]-ddec/2.,decs[-1]+ddec/2.])
    plt.gca().invert_xaxis()
    # if detector == "nrs1":
    #     plt.annotate("Simu comp 3uJy at {0}".format(photfilter_name),(0.4,-0.6),(0.0,-2),xycoords="data",color="red", fontsize=fontsize,
    #         arrowprops=dict(facecolor='red', shrink=0.05))
    # elif detector == "nrs2":
    #     plt.annotate("Simu comp 4uJy at {0}".format(photfilter_name),(0.4,-0.6),(0.0,-2),xycoords="data",color="red", fontsize=fontsize,
    #         arrowprops=dict(facecolor='red', shrink=0.05))
    plt.clim([-2,10])
    cbar = plt.colorbar()
    cbar.set_label("SNR",fontsize=fontsize)
    # plt.plot(out[:,0,0,2])
    plt.xlabel("dRA (as)",fontsize=fontsize)
    plt.ylabel("ddec (as)",fontsize=fontsize)
    plt.gca().tick_params(axis='x', labelsize=fontsize)
    plt.gca().tick_params(axis='y', labelsize=fontsize)
    plt.tight_layout()

    plt.tight_layout()
    out_filename = os.path.join(out_png,formatted_datetime+"_"+detector+"_combined.png")
    print("Saving " + out_filename)
    plt.savefig(out_filename)
    plt.savefig(out_filename.replace(".png",".pdf"))

    plt.figure(7,figsize=(5,4))
    hist, bins = np.histogram(np.concatenate(snr_vals), bins=20 * 3,range=(-10, 10))
    bin_centers = (bins[1::] + bins[0:np.size(bins) - 1]) / 2.
    plt.plot(bin_centers, hist / (np.nansum(hist) * (bins[1] - bins[0])), label="snr map")
    plt.plot(bin_centers, 1 / np.sqrt(2 * np.pi) * np.exp(-0.5 * (bin_centers - 0.0) ** 2), color="black",
             linestyle="--", label="Gaussian")
    plt.yscale("log")
    plt.xlim([-6, 6])
    plt.ylim([1e-4, 1])
    plt.xlabel("SNR",fontsize=fontsize)
    plt.ylabel("PDF",fontsize=fontsize)
    plt.legend()
    plt.gca().tick_params(axis='x', labelsize=fontsize)
    plt.gca().tick_params(axis='y', labelsize=fontsize)
    plt.tight_layout()
    out_filename = os.path.join(out_png,formatted_datetime+"_"+detector+"_histall.png")
    print("Saving " + out_filename)
    plt.savefig(out_filename)
    plt.savefig(out_filename.replace(".png",".pdf"))

    plt.figure(8,figsize=(5,4))
    plt.scatter(rs_grid,contrast_5sig_combined,alpha=1,s=0.2,label=photfilter_name+" NIRSpec FM")
    # seps_Jens = np.load("/stow/jruffio/data/JWST/nirspec/HD_19467/breads/external/ADI_NANNU1_NSUBS1_JWST_NIRCAM_NRCALONG_"+photfilter_name+"_MASKBAR_MASKALWB_SUB320ALWB-KLmodes-all_seps.npy")
    # cons_Jens = np.load("/stow/jruffio/data/JWST/nirspec/HD_19467/breads/external/ADI_NANNU1_NSUBS1_JWST_NIRCAM_NRCALONG_"+photfilter_name+"_MASKBAR_MASKALWB_SUB320ALWB-KLmodes-all_cons.npy")
    # plt.plot(seps_Jens[0],np.nanmin(cons_Jens,axis=0),label=photfilter_name+" KLIP Jens",alpha=1)
    plt.fill_between([0,0.5],[10**(-10),10**(-10)],[1,1],color="grey",alpha=0.2)
    plt.title("5-$\sigma$ Sensitivity 1D {0}".format(detector),fontsize=fontsize)
    plt.yscale("log")
    plt.xlim([0,3])
    if detector == "nrs1":
        plt.ylim([10**(-6.5),10**(-2)])
    elif detector == "nrs2":
        plt.ylim([10**(-6.5),10**(-2)])
    plt.xlabel("Separation (as)",fontsize=fontsize)
    plt.ylabel("5-$\sigma$ Flux ratio ({0})".format(photfilter_name),fontsize=fontsize)
    plt.scatter(rgrid_psfprofile.ravel(),im_psfprofile.ravel(),c="black",s=0.5,label="PSF profile")
    plt.legend(loc="upper left",frameon=True,fontsize=fontsize)
    plt.gca().tick_params(axis='x', labelsize=fontsize)
    plt.gca().tick_params(axis='y', labelsize=fontsize)
    plt.tight_layout()
    out_filename = os.path.join(out_png,formatted_datetime+"_"+detector+"_combinedcontrast.png")
    print("Saving " + out_filename)
    plt.savefig(out_filename)
    plt.savefig(out_filename.replace(".png",".pdf"))

    if 1:
        out_filename = os.path.join(out_png,"HD19467B_FM_"+photfilter_name+"_"+detector+"_hist.fits")
        print(out_filename)
        hdulist = fits.HDUList()
        hdulist.append(fits.PrimaryHDU(data=bin_centers))
        hdulist.append(fits.ImageHDU(data= hist / (np.nansum(hist) * (bins[1] - bins[0]))))
        try:
            hdulist.writeto(out_filename, overwrite=True)
        except TypeError:
            hdulist.writeto(out_filename, clobber=True)
        hdulist.close()

        out_filename = os.path.join(out_png,"HD19467B_FM_"+photfilter_name+"_"+detector+"_SNR.fits")
        hdulist = fits.HDUList()
        hdulist.append(fits.PrimaryHDU(data=snr_map_combined))
        try:
            hdulist.writeto(out_filename, overwrite=True)
        except TypeError:
            hdulist.writeto(out_filename, clobber=True)
        hdulist.close()

        out_filename = os.path.join(out_png,"HD19467B_FM_"+photfilter_name+"_"+detector+"_flux.fits")
        hdulist = fits.HDUList()
        hdulist.append(fits.PrimaryHDU(data=fluxmap_combined))
        try:
            hdulist.writeto(out_filename, overwrite=True)
        except TypeError:
            hdulist.writeto(out_filename, clobber=True)
        hdulist.close()

        out_filename = os.path.join(out_png,"HD19467B_FM_"+photfilter_name+"_"+detector+"_fluxerr.fits")
        hdulist = fits.HDUList()
        hdulist.append(fits.PrimaryHDU(data=fluxerrmap_combined))
        try:
            hdulist.writeto(out_filename, overwrite=True)
        except TypeError:
            hdulist.writeto(out_filename, clobber=True)
        hdulist.close()

        out_filename = os.path.join(out_png,"HD19467B_FM_"+photfilter_name+"_"+detector+"_fluxratioerr_1sig.fits")
        hdulist = fits.HDUList()
        hdulist.append(fits.PrimaryHDU(data=contrast_5sig_combined/5.))
        try:
            hdulist.writeto(out_filename, overwrite=True)
        except TypeError:
            hdulist.writeto(out_filename, clobber=True)
        hdulist.close()

        out_filename = os.path.join(out_png,"HD19467B_FM_"+photfilter_name+"_"+detector+"_RADec.fits")
        hdulist = fits.HDUList()
        hdulist.append(fits.PrimaryHDU(data=ras_grid))
        hdulist.append(fits.ImageHDU(data=decs_grid))
        try:
            hdulist.writeto(out_filename, overwrite=True)
        except TypeError:
            hdulist.writeto(out_filename, clobber=True)
        hdulist.close()


    plt.show()




    exit()