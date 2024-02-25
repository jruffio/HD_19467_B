import numpy as np
import matplotlib.pyplot as plt
from  scipy.interpolate import interp1d
import os
from astropy.table import Table

import matplotlib
matplotlib.use('Qt5Agg')

if __name__ == "__main__":
    try:
        import mkl
        mkl.set_num_threads(1)
    except:
        pass

    fontsize=12
    out_png = "/stow/jruffio/data/JWST/nirspec/HD_19467/breads/figures"
    dir_Jens_contrast = "/stow/jruffio/data/JWST/external/20240205_Jens_Nircam_HD19467B"
    color_list = ["#ff9900","#006699", "#6600ff", "pink", "grey", "black"]
    linestyle_list = ["-","--", "-.", ":", "-", "--"]

    # Absolute fluxes for the host star to be used in calculated flux ratios with the companion.
    HD19467_flux_MJy = {'F250M': 3.51e-06,  # in MJy, Ref Greenbaum+2023 plus some custom calculations for other filters
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
    HD19467B_Jens_flux_MJy = {'F250M': 1.55513197154851e-11, # in MJy, Jens Kammerer
                              'F300M': 3.15498605577005e-11,
                              'F360M': 6.24010659177986e-11,
                              'F410M': 1.87104128636645e-10,
                              'F430M': 1.25655947454195e-10,
                              'F460M': 8.442392966964669e-11}
    HD19467B_Jens_fluxerr_MJy = {'F250M': 1.08615690205716e-12, # in MJy, Jens Kammerer
                                 'F300M': 3.42717495852672e-13,
                                 'F360M': 4.890020815607521e-13,
                                 'F410M': 1.2785999497417e-12,
                                 'F430M': 1.11102406014966e-12,
                                 'F460M': 1.03190607430818e-12}

    # for photfilter_name in HD19467B_Jens_flux_MJy.keys():
    #     photfilter = "/stow/jruffio/data/JWST/nirspec/HD_19467/breads/external/JWST_NIRCam." + photfilter_name + ".dat"
    #     filter_arr = np.loadtxt(photfilter)
    #     trans_wvs = filter_arr[:, 0] / 1e4
    #     trans = filter_arr[:, 1]
    #     photfilter_f = interp1d(trans_wvs, trans, bounds_error=False, fill_value=0)
    #     photfilter_wv0 = np.nansum(trans_wvs * photfilter_f(trans_wvs)) / np.nansum(photfilter_f(trans_wvs))
    #     bandpass = np.where(photfilter_f(trans_wvs) / np.nanmax(photfilter_f(trans_wvs)) > 0.01)
    #     photfilter_wvmin, photfilter_wvmax = trans_wvs[bandpass[0][0]], trans_wvs[bandpass[0][-1]]
    #     print(photfilter_name, photfilter_wvmin, photfilter_wvmax)

    denominator = 0
    seps_5sig_Jens = np.arange(0,3,0.1)
    HD19467B_Jens_flux_MJy_keys = np.array([mykey for mykey in HD19467B_Jens_flux_MJy.keys()])[::-1]
    plt.figure(1,figsize=(6,5))
    for photfilter_name,color,linestyle in zip(HD19467B_Jens_flux_MJy_keys,color_list,linestyle_list):

        # Read the ecsv file
        table = Table.read(os.path.join(dir_Jens_contrast,"ADI_NANNU1_NSUBS1_JWST_NIRCAM_NRCALONG_"+photfilter_name+"_MASKBAR_MASKLWB_SUB320ALWB-KLmodes-all_contrast.ecsv"))
        print(table) #ADI_NANNU1_NSUBS1_JWST_NIRCAM_NRCALONG_F250M_MASKBAR_MASKLWB_SUB320ALWB-KLmodes-all_contrast.ecsv
        seps_Jens = np.array(table['separation'])
        extracted_data=[]
        for Nkl in [1,2,3,4,5,6,7,8,9,10,20,50,100]:
            extracted_data.append(table['contrast, N_kl={0}'.format(Nkl)])
        # Convert the extracted columns to a NumPy array
        cons_Jens = np.column_stack(extracted_data)

        plt.plot(seps_Jens,np.nanmin(cons_Jens,axis=1),label=photfilter_name,color=color,linestyle=linestyle)
        plt.scatter(1.6,HD19467B_Jens_flux_MJy[photfilter_name]/HD19467_flux_MJy[photfilter_name],c=color,marker="x")
        if photfilter_name == "F460M":
            plt.text(1.8,HD19467B_Jens_flux_MJy[photfilter_name]/HD19467_flux_MJy[photfilter_name], photfilter_name, fontsize=fontsize, ha='left', va='top',color=color)# \n 36-166 $\mu$Jy
        else:
            plt.text(1.8,HD19467B_Jens_flux_MJy[photfilter_name]/HD19467_flux_MJy[photfilter_name], photfilter_name, fontsize=fontsize, ha='left', va='center',color=color)# \n 36-166 $\mu$Jy

    plt.text(0.01, 0.01, 'NIRCam - HD 19467 B', fontsize=fontsize, ha='left', va='bottom', color="black",transform=plt.gca().transAxes)  # \n 36-166 $\mu$Jy
    plt.yscale("log")
    plt.xscale("log")
    plt.ylim([1e-7,10**(-3.5)])
    plt.xlim([1e-1,20])
    plt.legend(loc="upper right",fontsize=fontsize)
    plt.xlabel("Separation (as)",fontsize=fontsize)
    plt.ylabel("Flux ratio (5$\sigma$)",fontsize=fontsize)
    plt.legend(loc="upper right",frameon=True,fontsize=10)
    plt.gca().tick_params(axis='x', labelsize=fontsize)
    plt.gca().tick_params(axis='y', labelsize=fontsize)
    plt.tight_layout()

    out_filename = os.path.join(out_png, "NIRCam_contrast.png")
    print("Saving " + out_filename)
    plt.savefig(out_filename, dpi=300)
    plt.savefig(out_filename.replace(".png", ".pdf"))



    plt.show()

