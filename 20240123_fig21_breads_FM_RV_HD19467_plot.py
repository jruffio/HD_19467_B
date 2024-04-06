import numpy as np
import matplotlib.pyplot as plt
import os
from glob import glob
import h5py
from  scipy.interpolate import interp1d
from scipy.interpolate import RegularGridInterpolator
import astropy.units as u
from astropy import constants as const
import astropy.io.fits as fits

from breads.utils import get_err_from_posterior

import matplotlib
matplotlib.use('Qt5Agg')

import matplotlib.gridspec as gridspec # GRIDSPEC !
from matplotlib.colorbar import Colorbar
import matplotlib.patheffects as PathEffects
import matplotlib
matplotlib.rcParams.update({'errorbar.capsize': 2})


if __name__ == "__main__":
    try:
        import mkl
        mkl.set_num_threads(1)
    except:
        pass

    ####################
    ## To be modified
    ####################
    # directory where the RV grid fits are saved: the outputs from breads.grid_search
    out_dir = "/stow/jruffio/data/JWST/nirspec/HD_19467/breads/20240225_out_fm/RVs/"
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
    if not os.path.exists(out_png):
        os.makedirs(out_png)
    # Science data: List of stage 2 cal.fits files
    filelist = glob("/stow/jruffio/data/JWST/nirspec/HD_19467/HD19467_onaxis_roll2/20240124_stage2_clean/jw01414004001_02101_*_nrs*_cal.fits")
    filelist.sort()
    for filename in filelist:
        print(filename)
    print("N files: {0}".format(len(filelist)))
    color_list = ["#ff9900","#006699", "#6600ff", "#006699", "#ff9900", "#6600ff"]
    ####################

    plt.figure(1)
    if 1:
        RV_bestfit = np.zeros((2,9))+np.nan
        RV_errm = np.zeros((2,9))+np.nan
        RV_errp = np.zeros((2,9))+np.nan
        for detid, detector in enumerate(["nrs1","nrs2"]):
            filelist_nrs = [filename for filename in filelist if detector in filename]
            for fid,filename in enumerate(filelist_nrs):

                print(os.path.join(out_dir,"*_"+os.path.basename(filename).replace(".fits","_out.h5")))
                outoftheoven_filelist = glob(os.path.join(out_dir,"*_"+os.path.basename(filename).replace(".fits","_out.h5")))
                outoftheoven_filelist.sort()
                if len(outoftheoven_filelist)==0:
                    continue
                print(outoftheoven_filelist)
                outoftheoven_filename = outoftheoven_filelist[-1]
                with h5py.File(outoftheoven_filename, 'r') as hf:
                    rvs = np.array(hf.get("rvs"))
                    log_prob = np.array(hf.get("log_prob"))
                    log_prob_H0 = np.array(hf.get("log_prob_H0"))
                    rchi2 = np.array(hf.get("rchi2"))
                    linparas = np.array(hf.get("linparas"))
                    linparas_err = np.array(hf.get("linparas_err"))



                print(log_prob.shape)
                RV, _RVerr_m,_RVerr_p = get_err_from_posterior(rvs,np.exp(log_prob- np.nanmax(log_prob)))
                print(RV, _RVerr_m,_RVerr_p)
                RV_bestfit[detid,fid] = RV
                RV_errm[detid,fid] = _RVerr_m
                RV_errp[detid,fid] = _RVerr_p

                plt.plot(rvs,np.exp(log_prob- np.nanmax(log_prob)),label=detector+" {0}".format(fid))
    plt.legend()
    # plt.show()
    # exit()
    RVerr = np.nanmax(np.concatenate((RV_errp[:,:,None],RV_errm[:,:,None]),axis=2),axis=2)
    RV_bestfit_combined = np.nansum(RV_bestfit / RVerr ** 2, axis=1) / np.nansum(1 / RVerr ** 2, axis=1)
    print("RV_bestfit_combined",RV_bestfit_combined)
    RV_err_combined = np.sqrt(1/np.nansum(1/RVerr**2,axis=1))
    print("fluxMJy_err_combined",RV_err_combined)


    plt.figure(2,figsize=(6,5.5))
    for detid, detector in enumerate(["nrs1","nrs2"]):
    # for detector in ["nrs2"]:
        if detid ==0:
            plt.scatter(np.arange(RV_bestfit.shape[1])+1,RV_bestfit[detid,:],label=detector,marker="x")
        else:
            plt.scatter(np.arange(RV_bestfit.shape[1])+1,RV_bestfit[detid,:],label=detector,marker="o")
        plt.errorbar(np.arange(RV_bestfit.shape[1])+1,RV_bestfit[detid,:],yerr=[RV_errm[detid,:],RV_errp[detid,:]], ls='none')
        plt.ylim([-8,12])
        plt.xlabel("Dither #",fontsize=fontsize)
        plt.ylabel(r"RV (km/s)",fontsize=fontsize)
        plt.gca().tick_params(axis='x', labelsize=fontsize)
        plt.gca().tick_params(axis='y', labelsize=fontsize)
        plt.xticks([1,2,3,4,5,6,7,8,9])
        plt.legend(loc="upper left", handlelength=3,fontsize=fontsize)

    plt.text(0.01, 0.01, 'NOT corrected for systematics \n(e.g., NIRSpec wavelength calibration)', fontsize=fontsize, ha='left', va='bottom', color="black",
             transform=plt.gca().transAxes)
    # plt.tight_layout()
    out_filename = os.path.join(out_png, "HD19467B_RVs.png")
    print("Saving " + out_filename)
    plt.savefig(out_filename, dpi=300)
    plt.savefig(out_filename.replace(".png", ".pdf"))


    plt.show()



    exit()