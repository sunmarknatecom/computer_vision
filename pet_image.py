# Mark_PET_image.property
#
# Copyright (c) 2018 Mark S. Hong
#
# Release history
## Nov-21-2018: version 0.1
#   1) BMI addition
#   2) tqdm correction#
## Nov-22-2018: version 0.1.0.1
#   add function "def compare_two_dicoms():"
## Nov-23-2018: version 0.1.0.2
#   add function "def IQA():"
## Nov-26-2018: version 0.1.1
#    modify function "def IQA():"
#        : serial variables to list format
## Nov-27-2018: version 0.1.1.1
#
## Nov-29-2018: version 0.1.1.2
#    add function "def serial_rename():"
## Nov-29-2018: version 0.1.1.3
#    add function "def get_pt():"
## Nov-29-2018: version 0.1.2
#   add function "def serial_mse_ssim():"
## Dec-01-2018: version 0.1.3
#   add function "def serial_WB():"
#   mse and ssim between a previous image and a next image
## Dec-04-2018: version 0.1.4
#   add function "def stab_f():"
#   stabilization factor => SUVtst / SUVref
## Dec-04-2018: version 0.1.4.1
#   add function "def ssf():"
#   180s, 150s, 120s, 90s, 60s, 30s image sf acquisition
## Jan-04-2019: version 0.1.4.2
#   fix bug : def mssf() -> add_file_list variable fix var_list_3 -> var_list_2
## Jan-07-2019: version 0.1.5.0
#   add basic_stat()
#
#
# To do list : Done (Nov-23-2018)
#    I. Objective Image Assessment
#    o  1. MSE + PSNR
#    o  2. Structural similarity index (SSIM)
#    x  3. Multi-scale structural similarity index (MS-SSIM)
#    x  4. Visual information fidelity (VIF) = Source model + Distortion model + HVS model
#    x  5. Most apparent distortion (MAD)
#    x  6. Feature similarity index (FSIM)
# Reference)
# 
# To do list
#    I. def IQA() : automatication
#     II. patient information completion

import pydicom
import numpy as np
import matplotlib.pyplot as plt
import csv
import pandas as pd # exporting by pandas
from tqdm import tqdm # exporting by progress bar
import os
from skimage.measure import compare_ssim as ssim
import glob

# name input functions##########################################################
def get_filename():
    filename = input("Type the name of file >>>")
    return filename

def get_start_number():
    start_number = input("Type the start number >>>")
    return start_number

def get_stop_number():
    stop_number = input("Type the stop number >>>")
    return stop_number

def get_init_char():
    ini_chr = input("Type the initial character >>>")
    return ini_chr

def get_ext_char():
    ext_chr = input("Type the extension character >>>")
    return ext_chr

def get_name_output_file():
    output_file = input("Type the name of output file >>>")
    return output_file

def get_cf_of_a_file(fn):

    ds = pydicom.dcmread(fn)

    # body_weight        =    bdwt
    # tracer_activity    =    trac
    # series_time        =    seti
    # measure_time        =    meti
    # half_life            =    half
    # rescale_slope        =    resl

    bdwt = ds[0x0010,0x1030].value
    trac = ds[0x0054,0x0016][0][0x0018,0x1074].value
    seti = ds[0x0008,0x0032].value
    meti = ds[0x0054,0x0016][0][0x0018,0x1072].value
    half = ds[0x0054,0x0016][0][0x0018,0x1075].value
    resl = ds[0x0028,0x1053].value

    # 4th step: arrangement of variables

    st = seti[0:6]
    mt = meti[0:6]
    rs = float(resl)

    # hour_sec        = hs (시간을 초로 변환하기 위하여 3600 곱)
    # minute_sec    = ms (분을 초로 변환하기 위하여 60 곱)
    # second        = se
    # total_time    = tt (unit: seconds)

    hs = (int(st[0:2]) - int(mt[0:2])) * 3600
    ms = (int(st[2:4]) - int(mt[2:4])) * 60
    se = (int(st[4:]) - int(mt[4:]))
    tt = hs + ms + se

    # expo = exponential
    # acac = actual_activity

    expo = tt / half
    acac = trac * 2 ** (-1 * expo)

    # conversion_factor = cf

    cf= rs * bdwt * 1000 / acac

    return cf

# file-out as CSV
def csv_output(_fn, list1, list2):

    np.savetxt('%s_SUV_data.csv' %_fn, list1, delimiter=",")
    np.savetxt('%s_raw_data.csv'%_fn, list2, delimiter=",")

# image viewing by matplotlib, need function get_cf_of_a_file
# clist is the list of SUVs
def im_view_clist(fn):
    ds = pydicom.dcmread(fn)
    list = ds.pixel_array
    cf = get_cf_of_a_file(fn)
    clist = list * cf

    plt.imshow(clist, interpolation="bicubic", cmap="Greys")
    plt.colorbar()
    plt.show()

# first order functions#########################################################

# second function ##############################################################
# export converted to SUV data
def export_cdata_results():

    init_char = get_init_char()
    start_number = int(get_start_number())
    stop_number = int(get_stop_number()) + 1
    ext_char = get_ext_char()
    output_name_file = get_name_output_file()
    first_order_label = ["filenames", "Slice Number", "cf", "BMI", "min", "max", "25 percentile", "50 percentile", "75 percentile", "mean", "std", "var"]

    # output file making
    f = open(output_name_file, 'w', newline='')
    wr = csv.writer(f)
    wr.writerow(first_order_label)

    pbar = tqdm(range(start_number,stop_number))

    for i in pbar:

        pbar.set_description("Exporting : ")

        fns = "%s%03d.%s" %(init_char,i,ext_char)
        ds_name = pydicom.dcmread(fns)
        list = ds_name.pixel_array

#** genuine output data*********************************************************

        bdwt = ds_name[0x0010,0x1030].value
        trac = ds_name[0x0054,0x0016][0][0x0018,0x1074].value
        seti = ds_name[0x0008,0x0032].value
        meti = ds_name[0x0054,0x0016][0][0x0018,0x1072].value
        half = ds_name[0x0054,0x0016][0][0x0018,0x1075].value
        resl = ds_name[0x0028,0x1053].value

# 4th step: arrangement of variables

        st = seti[0:6]
        mt = meti[0:6]
        rs = float(resl)

# hour_sec        = hs (시간을 초로 변환하기 위하여 3600 곱)
# minute_sec    = ms (분을 초로 변환하기 위하여 60 곱)
# second        = se
# total_time    = tt (unit: seconds)

        hs = (int(st[0:2]) - int(mt[0:2])) * 3600
        ms = (int(st[2:4]) - int(mt[2:4])) * 60
        se = (int(st[4:]) - int(mt[4:]))
        tt = hs + ms + se

# expo = exponential
# acac = actual_activity

        expo = tt / half
        acac = trac * 2 ** (-1 * expo)

        # conversion_factor = cf

        cf= rs * bdwt * 1000 / acac
        converted_list = list * cf

# BMI calculation
        height = ds_name[0x0010,0x1020].value
        BMI = bdwt / (height * height)
        slice_no = ds_name[0x0020,0x1041].value

#** genuine output data*********************************************************
        # raw_data_first_order = [fns, slice_no, cf, BMI, np.min(list), np.max(list), np.percentile(list, 25), np.percentile(list, 50), np.percentile(list, 75), np.mean(list), np.std(list), np.nanvar(list)]
#** genuine output data*********************************************************
        converted_data_first_order = [fns, slice_no, cf, BMI, np.min(converted_list), np.max(converted_list), np.percentile(converted_list, 25), np.percentile(converted_list, 50), np.percentile(converted_list, 75), np.mean(converted_list), np.std(converted_list), np.nanvar(converted_list)]
#** genuine output data*********************************************************

#       print(raw_data_first_order)
#       print(converted_data_first_order)
# output - adding
#       wr.writerow(raw_data_first_order)
        wr.writerow(converted_data_first_order)
    f.close()

# 01 min
#   >>> np.min(list)
# 02 nanmax
#   >>> np.max(list)
# 03 percentile
#   >>> np.percentile(list, 50)
# 04 median
#   >>> np.median(list)
# 05 mean
#   >>> np.mean(list)
# 08 std
#   >>> np.std(list)
# 09 var
#   >>> np.var(list)

def export_list_csv():

    init_char = get_init_char()
    start_number = int(get_start_number())
    stop_number = int(get_stop_number()) + 1
    ext_char = get_ext_char()

    pbar = tqdm(range(start_number,stop_number))

    for i in pbar:

        pbar.set_description("Exporting : ")

        fns = "%s%03d.%s" %(init_char,i,ext_char)
        ds_name = pydicom.dcmread(fns)
        list = ds_name.pixel_array

#** genuine output data*********************************************************

        bdwt = ds_name[0x0010,0x1030].value
        trac = ds_name[0x0054,0x0016][0][0x0018,0x1074].value
        seti = ds_name[0x0008,0x0032].value
        meti = ds_name[0x0054,0x0016][0][0x0018,0x1072].value
        half = ds_name[0x0054,0x0016][0][0x0018,0x1075].value
        resl = ds_name[0x0028,0x1053].value

# 4th step: arrangement of variables

        st = seti[0:6]
        mt = meti[0:6]
        rs = float(resl)

# hour_sec        = hs (시간을 초로 변환하기 위하여 3600 곱)
# minute_sec    = ms (분을 초로 변환하기 위하여 60 곱)
# second        = se
# total_time    = tt (unit: seconds)

        hs = (int(st[0:2]) - int(mt[0:2])) * 3600
        ms = (int(st[2:4]) - int(mt[2:4])) * 60
        se = int(st[4:]) - int(mt[4:])
        tt = hs + ms + se

# expo = exponential
# acac = actual_activity

        expo = tt / half
        acac = trac * 2 ** (-1 * expo)

# conversion_factor = cf

        cf= rs * bdwt * 1000 / acac
        converted_list = list * cf

        np.savetxt('%s%03d_SUV_data.csv' %(init_char,i), converted_list, delimiter=",")

# csv 파일을 list 형식으로 읽기(pandas module)
# numpy만 사용시에는 import에서 pandas 생략
def read_csv_list_pandas():

    filename = get_name_output_file()
    df = pd.read_csv(filename, sep=',', header=None)
    return df

def read_cl_pandas(fn):
    df = pd.read_csv(fn, sep=',', header=0)
    return df

def read_csv_list_csv():

    filename = get_name_output_file()
    ds = np.loadtxt(filename, delimiter=',')
    return ds

def read_csv(fn):
    ds = np.loadtxt(fn, delimiter=',')
    return ds

def compare_two_dicoms():

#   f = IM-0002-0001.dcm
#   first, second = f[0:9]
#   start_number, last_number = f[9:12]
#   extention name = dcm

    first_set_files = input("first set file name ] ")
    second_set_files = input("second set file name ] ")
    start_number = input("start number ] ")
    last_number = input("last number ] ")
    ext_char =  input("extension character ] ")
    output_file_name = input("output filename ] ")

    cstart_number = int(start_number)
    clast_number = int(last_number) + 1

# output file making
    f = open(output_file_name, 'w', newline='')
    wr = csv.writer(f)

    pbar = tqdm(range(cstart_number,clast_number))

    for i in pbar:

        pbar.set_description("Exporting : ")

        fns_1 = "%s%03d.%s" %(first_set_files, i, ext_char)
        fns_2 = "%s%03d.%s" %(second_set_files, i, ext_char)

        first_ds = pydicom.dcmread(fns_1)
        second_ds = pydicom.dcmread(fns_2)

        first_slice_no = first_ds[0x0020,0x1041].value
        second_slice_no = second_ds[0x0020,0x1041].value

        if first_slice_no == second_slice_no:
            pass
        else:
            error_file_list = [fns_1, fns_2]
            wr.writerow(error_file_list)
    f.close()

# print basic information

def basic_stat():
    
    time_value = input("Type the time value (ex, 30s -> 030)>>> ")
    output_filename = input("Type the output filename >>> ")

    initial_list = ["CJY", "CMS", "CSB", "HSL", "KBL", "KHY", "KJH", "KSO", "KYB", "KYC", "LJS", "LMD", "LYS", "LYT", "NSW", "PKC", "PMJ", "PSY", "PYH", "RJK", "YHS"] # GE
    # initial_list = ["AES", "CJH", "CJY", "HDO", "JCS", "JSY", "KHY", "KJW", "KJY", "KMY", "LBS", "LCJ", "LKY", "LSD", "NYJ", "PYS", "SUJ", "SYS", "WJS", "YKD", "YYH"] # SIEMENS
    file_header = ["filenames", "Slice Number", "cf", "BMI", "min", "max", "25 percentile", "50 percentile", "75 percentile", "mean", "std", "var"]

    f = open(output_filename, 'w', newline='')
    wr = csv.writer(f)
    wr.writerow(file_header)

    for i in initial_list:

        files_1 = '%s_%s_*.dcm' %(i,time_value)
        file_list_1 = sorted(glob.glob(files_1))
        file_number_1 = len(file_list_1)

        for j in range(file_number_1):
            fns = '%s_%s_%04d.dcm' %(i,time_value,j+1)
            print(fns)

            ds_name = pydicom.dcmread(fns)
            list = ds_name.pixel_array

            bdwt = ds_name[0x0010,0x1030].value
            trac = ds_name[0x0054,0x0016][0][0x0018,0x1074].value
            seti = ds_name[0x0008,0x0032].value
            meti = ds_name[0x0054,0x0016][0][0x0018,0x1072].value
            half = ds_name[0x0054,0x0016][0][0x0018,0x1075].value
            resl = ds_name[0x0028,0x1053].value

            st = seti[0:6]
            mt = meti[0:6]
            rs = float(resl)

            hs = (int(st[0:2]) - int(mt[0:2])) * 3600
            ms = (int(st[2:4]) - int(mt[2:4])) * 60
            se = (int(st[4:]) - int(mt[4:]))
            tt = hs + ms + se

            expo = tt / half
            acac = trac * 2 ** (-1 * expo)

            cf= rs * bdwt * 1000 / acac
            converted_list = list * cf

            height = ds_name[0x0010,0x1020].value
            BMI = bdwt / (height * height)
            slice_no = ds_name[0x0020,0x1041].value

            converted_data_first_order = [fns, slice_no, cf, BMI, np.min(converted_list), np.max(converted_list), np.percentile(converted_list, 25), np.percentile(converted_list, 50), np.percentile(converted_list, 75), np.mean(converted_list), np.std(converted_list), np.nanvar(converted_list)]

            wr.writerow(converted_data_first_order)
            
    f.close()

# IQA
def IQA():
    phrases = ["original file name :", "first file name :", "second file name : ", "third file name : ", "fourth file name : ", "fifth file name : "]
    filename = list(range(6)); ds_list = list(range(6))
    list_list = list(range(6)); clist_list = list(range(6))
    mse_list = list(range(6)); ssim_list = list(range(6))
    bdwt_list = list(range(6)); trac_list = list(range(6))
    seti_list = list(range(6)); meti_list = list(range(6))
    half_list = list(range(6)); resl_list = list(range(6))
    st_list = list(range(6)); mt_list = list(range(6))
    rs_list = list(range(6)); hs_list = list(range(6))
    ms_list = list(range(6)); se_list = list(range(6))
    tt_list = list(range(6)); expo_list = list(range(6))
    acac_list = list(range(6)); cf_list = list(range(6))
    
    def mse(x, y):
        return np.linalg.norm(x - y)

    for i in range(1, 7):
        filename[i-1] = input(phrases[i-1])
        ds_list[i-1] = pydicom.dcmread(filename[i-1])
        list_list[i-1] = ds_list[i-1].pixel_array

        bdwt_list[i-1] = ds_list[i-1][0x0010,0x1030].value
        trac_list[i-1] = ds_list[i-1][0x0054,0x0016][0][0x0018,0x1074].value
        seti_list[i-1] = ds_list[i-1][0x0008,0x0032].value
        meti_list[i-1] = ds_list[i-1][0x0054,0x0016][0][0x0018,0x1072].value
        half_list[i-1] = ds_list[i-1][0x0054,0x0016][0][0x0018,0x1075].value
        resl_list[i-1] = ds_list[i-1][0x0028,0x1053].value

# 4th step: arrangement of variables

        st_list[i-1] = seti_list[i-1][0:6]
        mt_list[i-1] = meti_list[i-1][0:6]
        rs_list[i-1] = float(resl_list[i-1])

# hour_sec        = hs (시간을 초로 변환하기 위하여 3600 곱)
# minute_sec    = ms (분을 초로 변환하기 위하여 60 곱)
# second        = se
# total_time    = tt (unit: seconds)

        hs_list[i-1] = (int(st_list[i-1][0:2]) - int(mt_list[i-1][0:2])) * 3600
        ms_list[i-1] = (int(st_list[i-1][2:4]) - int(mt_list[i-1][2:4])) * 60
        se_list[i-1] = int(st_list[i-1][4:]) - int(mt_list[i-1][4:])
        tt_list[i-1] = hs_list[i-1] + ms_list[i-1] + se_list[i-1]

# expo = exponential
# acac = actual_activity

        expo_list[i-1] = tt_list[i-1] / half_list[i-1]
        acac_list[i-1] = trac_list[i-1] * 2 ** (-1 * expo_list[i-1])

# conversion_factor = cf

        cf_list[i-1] = rs_list[i-1] * bdwt_list[i-1] * 1000 / acac_list[i-1]
        clist_list[i-1] = list_list[i-1] * cf_list[i-1]

    # rows, cols = clist_list[0].shape

    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15,6), sharex=True, sharey=True)

    ax = axes.ravel()
    
    for i in range(1,7):

        mse_list[i-1] = mse(clist_list[0], clist_list[i-1])
        ssim_list[i-1] = ssim(clist_list[0], clist_list[i-1], data_range=clist_list[i-1].max() - clist_list[i-1].min())

    label = 'MSE: {:.2f}, SSIM: {:.2f}'

    ax_list = []
    ax_title = ['180s image', '150s image', '120s image', '90s image', '60s image', '30s image']
    for i in range(1,7):
        ax[i-1].imshow(clist_list[i-1], interpolation="bicubic", cmap="Greys")
        ax[i-1].set_xlabel(label.format((mse_list[i-1]), ssim_list[i-1]))
        ax[i-1].set_title(ax_title[i-1])

    plt.tight_layout()
    plt.show()

def cad(fn):
    ds = pydicom.dcmread(fn)
    at = ds[0x0009,0x106d].value
    return at

def serial_IQA():
    for i in range(1, 6):
        filename = input("filename_%d >>>" %i)
        ds = pydicom.dcmread(filename)
        list = ds.pixel_array
        print(list)

def create_file_list_in_dir():

    f_list = []

    for (path, dirname, filename) in os.walk(os.getcwd()):
        for files in filename:
            if files.endswith(".csv"):
                f_list.append(files)
    return f_list

def fusion_list():
    file_list = create_file_list_in_dir()
    combined_csv = pd.concat([pd.read_csv(f) for f in file_list])
    combined_csv.to_csv("output.csv", index=False)
# output_list.to_csv("output.csv", mode='a', header=False)
# time_pd2.to_csv("filename.csv", mode='a', header=False)

def serial_rename():
#    folder_name = os.getcwd()
#    file_list = os.listdir(folder_name)
    initial_name = input("Initial name? >>> ")
    files = '*.dcm'
    file_list2 = glob.glob(files)
    count_files = len(file_list2)
    acqusition_time = []
#   create ds list
    ds_list = []

    pbar = tqdm(range(count_files))
    for i in pbar:

        pbar.set_description("Processing (1/3) : ")
        
        ds_list.append(0)

    pbar = tqdm(range(count_files))
    for i in pbar:

        pbar.set_description("Processing (2/3) : ")

        filename = file_list2[i]
        ds_list[i] = pydicom.dcmread(filename)
        acqusition_time.append(ds_list[i][0x0009,0x106d].value)
    
    pbar = tqdm(range(count_files))
    for i in pbar:

        pbar.set_description("Processing (3/3) : ")

        type_at = "%03d" %acqusition_time[i]
        new_name = initial_name + '_' + type_at + '_' + file_list2[i][8:]
        os.rename(file_list2[i],new_name)

# patient information
def get_pt():
    
    files = '*.dcm'
    file_list = glob.glob(files)
    file_header = ["filename", "No.", "Sex", "Age", "Height(cm)", "Weight(kg)", "BMI(kg/m2)", "Inj. Dose(MBq)", "Start of measurement(min)"]
    
    output_filename = input("Type the filename of output? >>> ")
    
    f = open(output_filename, 'w', newline='')
    wr = csv.writer(f)
    wr.writerow(file_header)

    pbar = tqdm(range(len(file_list)))
    
    for i in pbar:

        pbar.set_description("Exporting : ")    
        ds = pydicom.dcmread(file_list[i])

        sex = ds[0x0010,0x0040].value
        age = (ds[0x0010,0x1010].value)[1:3]
        ht = float(ds[0x0010,0x1020].value)
        htcm = ht * 100
        wt = float(ds[0x0010,0x1030].value)
        BMI = wt / (ht * ht)
        indo = ds[0x0054,0x0016][0][0x0018,0x1074].value / 1000000
        # acqt = (ds[0x0009,0x106c].value)[8:14] # GE style
        acqt = (ds[0x0008,0x0032].value)[:6] # SIEMENS style
        # SIEMENS acquisition time => [0x0008,0x0032] 110750.000000 
        # injt = (ds[0x0009,0x103d].value)[8:14] # GE style
        injt = (ds[0x0054,0x0016][0][0x0018,0x1072].value)[:6] # SIEMENS style
        # SIEMENS injection time => [0x0054,0x0016][0][0x0018,0x1072] 
        measure_time = (int(acqt[:2])-int(injt[:2])) * 60 + (int(acqt[2:4])-int(injt[2:4]))
    
        converted_file_content = [file_list[i], i+1, sex, age, htcm, wt, "%02.1f" %BMI, int(indo), measure_time]

        wr.writerow(converted_file_content)

    f.close()

#* filename
#* No.
#* sex(Sex) = ds[0x0010,0x0040].value
#* age(Age) = ds[0x0010,0x1010].value
#* ht(Height) = float(ds[0x0010,0x1020].value)
#* wt(Weight) = float(ds[0x0010,0x1030].value)
#* BMI = wt / (ht * ht)
#* indo (injection dose) = ds[0x0054,0x0016][0][0x0018,0x1074].value
#     Start of time = acqusition time - inj time
# acqt(acqusition time) = (ds[0x0009,0x106c].value)[8:14]
# injt(injection time) = (ds[0x0009,0x103d].value)[8:14]
# 
#* measure_time(corrected acqusition time) = (int(acqt[:2])-int(injt[:2])) * 3600 + (int(acqt[2:4])-int(injt[2:4])) * 60 + (int(acqt[4:])-int(injt[4:]))
# acqt_h = int(acqt[:2])
# injt_h = int(injt[:2])
# 
# acqt_m = int(acqt[2:4])
# injt_m = int(injt[2:4])
# 
# acqt_s = int(acqt[4:])
# injt_s = int(acqt[4:])
#    print(file_list)

def serial_mse_ssim():

    output_filename = input("what is the output filename? >>>")
    
    initial_list = ["CJY", "CMS", "CSB", "HSL", "KBL", "KHY", "KJH", "KSO", "KYB", "KYC", "LJS", "LMD", "LYS", "LYT", "NSW", "PKC", "PMJ", "PSY", "PYH", "RJK", "YHS"]
    file_header = ["File Name", "mse-180s", "ssim-180s", "mse-150s", "ssim-150s", "mse-120s", "ssim-120s", "mse-90s", "ssim-90s", "mse-60s", "ssim-60s", "mse-30s", "ssim-30s"] 

    f = open(output_filename, 'w', newline='')
    wr = csv.writer(f)
    wr.writerow(file_header)
    
    def mse(x, y):
        return np.linalg.norm(x - y)
    
    for i in initial_list:

        files_1 = '%s_180_*.dcm' %i
        file_list_1 = sorted(glob.glob(files_1))
        file_number_1 = len(file_list_1)
        
        for j in range(file_number_1):
            files = '%s_*_%04d.dcm' %(i, j+1)
            file_list_2 = sorted(glob.glob(files)) # sorting of filename due to random order
            file_list_2.reverse()
            
            print(file_list_2)
            
            file_list_2[0] # %s_180_%04d.dcm
            file_list_2[1] # %s_150_%04d.dcm
            file_list_2[2] # %s_120_%04d.dcm
            file_list_2[3] # %s_090_%04d.dcm
            file_list_2[4] # %s_060_%04d.dcm
            file_list_2[5] # %s_030_%04d.dcm

            ds_tf_1 = pydicom.dcmread(file_list_2[0])
            ds_tf_2 = pydicom.dcmread(file_list_2[1])
            ds_tf_3 = pydicom.dcmread(file_list_2[2])
            ds_tf_4 = pydicom.dcmread(file_list_2[3])
            ds_tf_5 = pydicom.dcmread(file_list_2[4])
            ds_tf_6 = pydicom.dcmread(file_list_2[5])
            
            list_tf_1 = ds_tf_1.pixel_array
            list_tf_2 = ds_tf_2.pixel_array
            list_tf_3 = ds_tf_3.pixel_array
            list_tf_4 = ds_tf_4.pixel_array
            list_tf_5 = ds_tf_5.pixel_array
            list_tf_6 = ds_tf_6.pixel_array
            
            cf_tf_1 = get_cf_of_a_file(file_list_2[0])            
            cf_tf_2 = get_cf_of_a_file(file_list_2[1])
            cf_tf_3 = get_cf_of_a_file(file_list_2[2])
            cf_tf_4 = get_cf_of_a_file(file_list_2[3])            
            cf_tf_5 = get_cf_of_a_file(file_list_2[4])
            cf_tf_6 = get_cf_of_a_file(file_list_2[5])
                
            clist_tf_1 = cf_tf_1 * list_tf_1
            clist_tf_2 = cf_tf_2 * list_tf_2
            clist_tf_3 = cf_tf_3 * list_tf_3
            clist_tf_4 = cf_tf_4 * list_tf_4
            clist_tf_5 = cf_tf_5 * list_tf_5
            clist_tf_6 = cf_tf_6 * list_tf_6
            
            mse_180s = mse(clist_tf_1, clist_tf_1)
            mse_150s = mse(clist_tf_1, clist_tf_2)
            mse_120s = mse(clist_tf_1, clist_tf_3)
            mse_90s = mse(clist_tf_1, clist_tf_4)
            mse_60s = mse(clist_tf_1, clist_tf_5)
            mse_30s = mse(clist_tf_1, clist_tf_6)
            
            ssim_180s = ssim(clist_tf_1, clist_tf_1, data_range=clist_tf_1.max() - clist_tf_1.min())
            ssim_150s = ssim(clist_tf_1, clist_tf_2, data_range=clist_tf_2.max() - clist_tf_2.min())
            ssim_120s = ssim(clist_tf_1, clist_tf_3, data_range=clist_tf_3.max() - clist_tf_3.min())
            ssim_90s = ssim(clist_tf_1, clist_tf_4, data_range=clist_tf_4.max() - clist_tf_4.min())
            ssim_60s = ssim(clist_tf_1, clist_tf_5, data_range=clist_tf_5.max() - clist_tf_5.min())
            ssim_30s = ssim(clist_tf_1, clist_tf_6, data_range=clist_tf_6.max() - clist_tf_6.min())
            
            add_file_list = [file_list_2[0], mse_180s, ssim_180s, mse_150s, ssim_150s, mse_120s, ssim_120s, mse_90s, ssim_90s, mse_60s, ssim_60s, mse_30s, ssim_30s]
            
            wr.writerow(add_file_list)

    f.close()
            
#     files = '%s_%03d_*.dcm' %(initial_list[0], acqusition_time[0])
    
    


# target_list = []

# os.rename(target, new_name)

# ds[0x0009,0x106d].value

# rename rule : initial_name + acquisition_time + serinal_number[8:]

def serial_WB():
    file_name = input("What is the initial file name ? >>")
    last_number = input("What is the last number? >>")
    ouput_filename = input("What is the output filename? >>>")
    extension = ".dcm"
    file_header = ["filename1", "filename2", "mse", "ssim"]

    f = open(ouput_filename, 'w', newline='')
    wr = csv.writer(f)
    wr.writerow(file_header)

    def mse(x, y):
        return np.linalg.norm(x - y)

    fn_list = list(range(int(last_number)))
    ds_list = list(range(int(last_number)))
    ls_list = list(range(int(last_number)))
    mse_list = list(range(int(last_number)))
    ssim_list = list(range(int(last_number)))
    cf_list = list(range(int(last_number)))
    cls_list = list(range(int(last_number)))
    results_list = list(range(int(last_number)-1))

    pbar = tqdm(range(int(last_number)))

    for i in pbar:

        pbar.set_description("Processing : ")    

        fn_list[i] = "%s%03i%s" %(file_name, i+1, extension)
        ds_list[i] = pydicom.dcmread(fn_list[i])
        ls_list[i] = ds_list[i].pixel_array
        cf_list[i] = get_cf_of_a_file(fn_list[i])
        cls_list[i] = ls_list[i] * cf_list[i]

    pbar_2 = tqdm(range(int(last_number)-1))

    for i in pbar_2:

        pbar_2.set_description("Processing : ")

        mse_list[i] = mse(cls_list[i], cls_list[i+1])
        ssim_list[i] = ssim(cls_list[i], cls_list[i+1], data_range=cls_list[i+1].max() - cls_list[i+1].min())
        results_list[i] = [fn_list[i], fn_list[i+1], mse_list[i], ssim_list[i]]
        
        wr.writerow(results_list[i])
    f.close()
# in one folder, serial images --> mse and ssim between adjacent two images.

#******************************************************************************
# add function, stabilization factor => stab_f
# reference) Using SUV as a Guide to 18F-FDG Dose Reduction. JNM 2014; 55: 1998-2002
# a single file

def test_sf():
    output_filename = input("Type the output file name >>>")
    fn_list = ["LCJ_030_0064.dcm", "LCJ_060_0064.dcm", "LCJ_090_0064.dcm", "LCJ_120_0064.dcm", "LCJ_150_0064.dcm", "LCJ_180_0064.dcm"]
    ds_list = list(range(6))
    list_list = list(range(6))
    cf_list = list(range(6))
    clist_list = list(range(6))
    sf_list = list(range(6))

    file_header = ["filename", "count of non-zero", "average of sf", "std of sf", "var of sf"]

    f = open(output_filename, 'w', newline='')
    wr = csv.writer(f)
    wr.writerow(file_header)

    for i in range(6):
        ds_list[i] = pydicom.dcmread(fn_list[i])
        list_list[i] = ds_list[i].pixel_array
        cf_list[i] = get_cf_of_a_file(fn_list[i])
        clist_list[i] = list_list[i] * cf_list[i]

    for i in range(6):
        sf_list[i] = np.divide(clist_list[5-i], clist_list[5], out=np.zeros_like(clist_list[5-i]), where=clist_list[5]!=0)

    nonzero_list = list(range(6))
    nonzerocount_list = list(range(6))
    avg_list = list(range(6))
    std_list = list(range(6))
    var_list = list(range(6))
    results_list = list(range(6))

    for i in range(6):
        nonzerocount_list[i] = np.count_nonzero(sf_list[i])
        sf_list[i][sf_list[i] == 0 ] = np.nan
        avg_list[i] = np.nanmean(sf_list[i])
        std_list[i] = np.nanstd(sf_list[i])
        var_list[i] = np.nanvar(sf_list[i])

        # fn_list[5-i] / nonzerocount_list[i]
        results_list[i] = [fn_list[5-i], nonzerocount_list[i], avg_list[i], std_list[i], var_list[i]]

        wr.writerow(results_list[i])
    f.close()

#******** making figure
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15,6), sharex=True, sharey=True)
    ax = axes.ravel()
    ax_list = list(range(6))
    ax_title = ['180s image', '150s image', '120s image', '90s image', '60s image', '30s image']
    for i in range(6):
        ax[i].imshow(sf_list[i], interpolation="bicubic", cmap="Greys")
        ax[i].set_title(ax_title[i])

    plt.tight_layout()
    plt.show()
#^^^^^^^^^ making figure

#******************************************************************************
# serial files ssf() = serial_stabilization_factor()
#******************************************************************************

def ssf():

    output_filename = input("what is the output filename? >>>")
    
    initial_list = ["CJY", "CMS", "CSB", "HSL", "KBL", "KHY", "KJH", "KSO", "KYB", "KYC", "LJS", "LMD", "LYS", "LYT", "NSW", "PKC", "PMJ", "PSY", "PYH", "RJK", "YHS"]
    file_header = ["File Name", "mse-180s", "ssim-180s", "non_zero_cts_180s", "avg_sf-180s", "std_sf-180s", "var_sf-180s", "mse-150s", "ssim-150s", "non_zero_cts_150s", "avg_sf-150s", "std_sf-150s", "var_sf-150s", "mse-120s", "ssim-120s", "non_zero_cts_120s", "avg_sf-120s", "std_sf-120s", "var_sf-120s", "mse-90s", "ssim-90s", "non_zero_cts_90s", "avg_sf-90s", "std_sf-90s", "var_sf-90s", "mse-60s", "ssim-60s", "non_zero_cts_60s", "avg_sf-60s", "std_sf-60s", "var_sf-60s", "mse-30s", "ssim-30s", "non_zero_cts_30s", "avg_sf-30s", "std_sf-30s", "var_sf-30s"]

    f = open(output_filename, 'w', newline='')
    wr = csv.writer(f)
    wr.writerow(file_header)
    
    def mse(x, y):
        return np.linalg.norm(x - y)
    
    for i in initial_list:

        files_1 = '%s_180_*.dcm' %i
        file_list_1 = sorted(glob.glob(files_1))
        file_number_1 = len(file_list_1)
        
        for j in range(file_number_1):
            files = '%s_*_%04d.dcm' %(i, j+1)
            file_list_2 = sorted(glob.glob(files)) # sorting of filename due to random order
            file_list_2.reverse()
            
            print(file_list_2)
            
            file_list_2[0] # %s_180_%04d.dcm
            file_list_2[1] # %s_150_%04d.dcm
            file_list_2[2] # %s_120_%04d.dcm
            file_list_2[3] # %s_090_%04d.dcm
            file_list_2[4] # %s_060_%04d.dcm
            file_list_2[5] # %s_030_%04d.dcm

            ds_tf_1 = pydicom.dcmread(file_list_2[0])
            ds_tf_2 = pydicom.dcmread(file_list_2[1])
            ds_tf_3 = pydicom.dcmread(file_list_2[2])
            ds_tf_4 = pydicom.dcmread(file_list_2[3])
            ds_tf_5 = pydicom.dcmread(file_list_2[4])
            ds_tf_6 = pydicom.dcmread(file_list_2[5])
            
            list_tf_1 = ds_tf_1.pixel_array
            list_tf_2 = ds_tf_2.pixel_array
            list_tf_3 = ds_tf_3.pixel_array
            list_tf_4 = ds_tf_4.pixel_array
            list_tf_5 = ds_tf_5.pixel_array
            list_tf_6 = ds_tf_6.pixel_array
            
            cf_tf_1 = get_cf_of_a_file(file_list_2[0])            
            cf_tf_2 = get_cf_of_a_file(file_list_2[1])
            cf_tf_3 = get_cf_of_a_file(file_list_2[2])
            cf_tf_4 = get_cf_of_a_file(file_list_2[3])            
            cf_tf_5 = get_cf_of_a_file(file_list_2[4])
            cf_tf_6 = get_cf_of_a_file(file_list_2[5])
                
            clist_tf_1 = cf_tf_1 * list_tf_1
            clist_tf_2 = cf_tf_2 * list_tf_2
            clist_tf_3 = cf_tf_3 * list_tf_3
            clist_tf_4 = cf_tf_4 * list_tf_4
            clist_tf_5 = cf_tf_5 * list_tf_5
            clist_tf_6 = cf_tf_6 * list_tf_6
            
            mse_180s = mse(clist_tf_1, clist_tf_1)
            mse_150s = mse(clist_tf_1, clist_tf_2)
            mse_120s = mse(clist_tf_1, clist_tf_3)
            mse_90s = mse(clist_tf_1, clist_tf_4)
            mse_60s = mse(clist_tf_1, clist_tf_5)
            mse_30s = mse(clist_tf_1, clist_tf_6)
            
            ssim_180s = ssim(clist_tf_1, clist_tf_1, data_range=clist_tf_1.max() - clist_tf_1.min())
            ssim_150s = ssim(clist_tf_1, clist_tf_2, data_range=clist_tf_2.max() - clist_tf_2.min())
            ssim_120s = ssim(clist_tf_1, clist_tf_3, data_range=clist_tf_3.max() - clist_tf_3.min())
            ssim_90s = ssim(clist_tf_1, clist_tf_4, data_range=clist_tf_4.max() - clist_tf_4.min())
            ssim_60s = ssim(clist_tf_1, clist_tf_5, data_range=clist_tf_5.max() - clist_tf_5.min())
            ssim_30s = ssim(clist_tf_1, clist_tf_6, data_range=clist_tf_6.max() - clist_tf_6.min())
            
# sf_list_1 --> 180s/180s,  sf_list_2 --> 150s/180s, sf_list_3 --> 120s/180s, sf_list_4 --> 90s/180s, sf_list_5 --> 60s/180s, sf_list_6 --> 30s/180s

            sf_list_1 = np.divide(clist_tf_1, clist_tf_1, out=np.zeros_like(clist_tf_1), where=clist_tf_1!=0)
            sf_list_2 = np.divide(clist_tf_2, clist_tf_1, out=np.zeros_like(clist_tf_2), where=clist_tf_1!=0)
            sf_list_3 = np.divide(clist_tf_3, clist_tf_1, out=np.zeros_like(clist_tf_3), where=clist_tf_1!=0)
            sf_list_4 = np.divide(clist_tf_4, clist_tf_1, out=np.zeros_like(clist_tf_4), where=clist_tf_1!=0)
            sf_list_5 = np.divide(clist_tf_5, clist_tf_1, out=np.zeros_like(clist_tf_5), where=clist_tf_1!=0)
            sf_list_6 = np.divide(clist_tf_6, clist_tf_1, out=np.zeros_like(clist_tf_6), where=clist_tf_1!=0)

            nonzerocount_list_1 = np.count_nonzero(sf_list_1)
            nonzerocount_list_2 = np.count_nonzero(sf_list_2)
            nonzerocount_list_3 = np.count_nonzero(sf_list_3)
            nonzerocount_list_4 = np.count_nonzero(sf_list_4)
            nonzerocount_list_5 = np.count_nonzero(sf_list_5)
            nonzerocount_list_6 = np.count_nonzero(sf_list_6)

            sf_list_1[sf_list_1 == 0] = np.nan
            sf_list_2[sf_list_2 == 0] = np.nan
            sf_list_3[sf_list_3 == 0] = np.nan
            sf_list_4[sf_list_4 == 0] = np.nan
            sf_list_5[sf_list_5 == 0] = np.nan
            sf_list_6[sf_list_6 == 0] = np.nan

            avg_list_1 = np.nanmean(sf_list_1)
            avg_list_2 = np.nanmean(sf_list_2)
            avg_list_3 = np.nanmean(sf_list_3)
            avg_list_4 = np.nanmean(sf_list_4)
            avg_list_5 = np.nanmean(sf_list_5)
            avg_list_6 = np.nanmean(sf_list_6)

            std_list_1 = np.nanstd(sf_list_1)
            std_list_2 = np.nanstd(sf_list_2)
            std_list_3 = np.nanstd(sf_list_3)
            std_list_4 = np.nanstd(sf_list_4)
            std_list_5 = np.nanstd(sf_list_5)
            std_list_6 = np.nanstd(sf_list_6)

            var_list_1 = np.nanvar(sf_list_1)
            var_list_2 = np.nanvar(sf_list_2)
            var_list_3 = np.nanvar(sf_list_3)
            var_list_4 = np.nanvar(sf_list_4)
            var_list_5 = np.nanvar(sf_list_5)
            var_list_6 = np.nanvar(sf_list_6)

            add_file_list = [file_list_2[0], mse_180s, ssim_180s, nonzerocount_list_1, avg_list_1, std_list_1, var_list_1, mse_150s, ssim_150s, nonzerocount_list_2, avg_list_2, std_list_2, var_list_2, mse_120s, ssim_120s, nonzerocount_list_3, avg_list_3, std_list_3, var_list_3, mse_90s, ssim_90s, nonzerocount_list_4, avg_list_4, std_list_4, var_list_4, mse_60s, ssim_60s, nonzerocount_list_5, avg_list_5, std_list_5, var_list_5, mse_30s, ssim_30s, nonzerocount_list_6, avg_list_6, std_list_6, var_list_6]
            
            wr.writerow(add_file_list)

    f.close()

def mssf():

    output_filename = input("what is the output filename? >>>")
    
    initial_list = ["AES", "CJH", "CJY", "HDO", "JCS", "JSY", "KHY", "KJW", "KJY", "KMY", "LBS", "LCJ", "LKY", "LSD", "NYJ", "PYS", "SUJ", "SYS", "WJS", "YKD", "YYH"]
    file_header = ["File Name", "mse-180s", "ssim-180s", "non_zero_cts_180s", "avg_sf-180s", "std_sf-180s", "var_sf-180s", "mse-150s", "ssim-150s", "non_zero_cts_150s", "avg_sf-150s", "std_sf-150s", "var_sf-150s", "mse-120s", "ssim-120s", "non_zero_cts_120s", "avg_sf-120s", "std_sf-120s", "var_sf-120s", "mse-90s", "ssim-90s", "non_zero_cts_90s", "avg_sf-90s", "std_sf-90s", "var_sf-90s", "mse-60s", "ssim-60s", "non_zero_cts_60s", "avg_sf-60s", "std_sf-60s", "var_sf-60s", "mse-30s", "ssim-30s", "non_zero_cts_30s", "avg_sf-30s", "std_sf-30s", "var_sf-30s"]

    f = open(output_filename, 'w', newline='')
    wr = csv.writer(f)
    wr.writerow(file_header)
    
    def mse(x, y):
        return np.linalg.norm(x - y)
    
    for i in initial_list:

        files_1 = '%s_180_*.dcm' %i
        file_list_1 = sorted(glob.glob(files_1))
        file_number_1 = len(file_list_1)
        
        for j in range(file_number_1):
            files = '%s_*_%04d.dcm' %(i, j+1)
            file_list_2 = sorted(glob.glob(files)) # sorting of filename due to random order
            file_list_2.reverse()
            
            print(file_list_2)
            
            file_list_2[0] # %s_180_%04d.dcm
            file_list_2[1] # %s_150_%04d.dcm
            file_list_2[2] # %s_120_%04d.dcm
            file_list_2[3] # %s_090_%04d.dcm
            file_list_2[4] # %s_060_%04d.dcm
            file_list_2[5] # %s_030_%04d.dcm

            ds_tf_1 = pydicom.dcmread(file_list_2[0])
            ds_tf_2 = pydicom.dcmread(file_list_2[1])
            ds_tf_3 = pydicom.dcmread(file_list_2[2])
            ds_tf_4 = pydicom.dcmread(file_list_2[3])
            ds_tf_5 = pydicom.dcmread(file_list_2[4])
            ds_tf_6 = pydicom.dcmread(file_list_2[5])
            
            list_tf_1 = ds_tf_1.pixel_array
            list_tf_2 = ds_tf_2.pixel_array
            list_tf_3 = ds_tf_3.pixel_array
            list_tf_4 = ds_tf_4.pixel_array
            list_tf_5 = ds_tf_5.pixel_array
            list_tf_6 = ds_tf_6.pixel_array
            
            cf_tf_1 = get_cf_of_a_file(file_list_2[0])            
            cf_tf_2 = get_cf_of_a_file(file_list_2[1])
            cf_tf_3 = get_cf_of_a_file(file_list_2[2])
            cf_tf_4 = get_cf_of_a_file(file_list_2[3])            
            cf_tf_5 = get_cf_of_a_file(file_list_2[4])
            cf_tf_6 = get_cf_of_a_file(file_list_2[5])
                
            clist_tf_1 = cf_tf_1 * list_tf_1
            clist_tf_2 = cf_tf_2 * list_tf_2
            clist_tf_3 = cf_tf_3 * list_tf_3
            clist_tf_4 = cf_tf_4 * list_tf_4
            clist_tf_5 = cf_tf_5 * list_tf_5
            clist_tf_6 = cf_tf_6 * list_tf_6
            
            mse_180s = mse(clist_tf_1, clist_tf_1)
            mse_150s = mse(clist_tf_1, clist_tf_2)
            mse_120s = mse(clist_tf_1, clist_tf_3)
            mse_90s = mse(clist_tf_1, clist_tf_4)
            mse_60s = mse(clist_tf_1, clist_tf_5)
            mse_30s = mse(clist_tf_1, clist_tf_6)
            
            ssim_180s = ssim(clist_tf_1, clist_tf_1, data_range=clist_tf_1.max() - clist_tf_1.min())
            ssim_150s = ssim(clist_tf_1, clist_tf_2, data_range=clist_tf_2.max() - clist_tf_2.min())
            ssim_120s = ssim(clist_tf_1, clist_tf_3, data_range=clist_tf_3.max() - clist_tf_3.min())
            ssim_90s = ssim(clist_tf_1, clist_tf_4, data_range=clist_tf_4.max() - clist_tf_4.min())
            ssim_60s = ssim(clist_tf_1, clist_tf_5, data_range=clist_tf_5.max() - clist_tf_5.min())
            ssim_30s = ssim(clist_tf_1, clist_tf_6, data_range=clist_tf_6.max() - clist_tf_6.min())
            
# sf_list_1 --> 180s/180s,  sf_list_2 --> 150s/180s, sf_list_3 --> 120s/180s, sf_list_4 --> 90s/180s, sf_list_5 --> 60s/180s, sf_list_6 --> 30s/180s

            sf_list_1 = np.divide(clist_tf_1, clist_tf_1, out=np.zeros_like(clist_tf_1), where=clist_tf_1!=0)
            sf_list_2 = np.divide(clist_tf_2, clist_tf_1, out=np.zeros_like(clist_tf_2), where=clist_tf_1!=0)
            sf_list_3 = np.divide(clist_tf_3, clist_tf_1, out=np.zeros_like(clist_tf_3), where=clist_tf_1!=0)
            sf_list_4 = np.divide(clist_tf_4, clist_tf_1, out=np.zeros_like(clist_tf_4), where=clist_tf_1!=0)
            sf_list_5 = np.divide(clist_tf_5, clist_tf_1, out=np.zeros_like(clist_tf_5), where=clist_tf_1!=0)
            sf_list_6 = np.divide(clist_tf_6, clist_tf_1, out=np.zeros_like(clist_tf_6), where=clist_tf_1!=0)

            nonzerocount_list_1 = np.count_nonzero(sf_list_1)
            nonzerocount_list_2 = np.count_nonzero(sf_list_2)
            nonzerocount_list_3 = np.count_nonzero(sf_list_3)
            nonzerocount_list_4 = np.count_nonzero(sf_list_4)
            nonzerocount_list_5 = np.count_nonzero(sf_list_5)
            nonzerocount_list_6 = np.count_nonzero(sf_list_6)

            sf_list_1[sf_list_1 == 0] = np.nan
            sf_list_2[sf_list_2 == 0] = np.nan
            sf_list_3[sf_list_3 == 0] = np.nan
            sf_list_4[sf_list_4 == 0] = np.nan
            sf_list_5[sf_list_5 == 0] = np.nan
            sf_list_6[sf_list_6 == 0] = np.nan

            avg_list_1 = np.nanmean(sf_list_1)
            avg_list_2 = np.nanmean(sf_list_2)
            avg_list_3 = np.nanmean(sf_list_3)
            avg_list_4 = np.nanmean(sf_list_4)
            avg_list_5 = np.nanmean(sf_list_5)
            avg_list_6 = np.nanmean(sf_list_6)

            std_list_1 = np.nanstd(sf_list_1)
            std_list_2 = np.nanstd(sf_list_2)
            std_list_3 = np.nanstd(sf_list_3)
            std_list_4 = np.nanstd(sf_list_4)
            std_list_5 = np.nanstd(sf_list_5)
            std_list_6 = np.nanstd(sf_list_6)

            var_list_1 = np.nanvar(sf_list_1)
            var_list_2 = np.nanvar(sf_list_2)
            var_list_3 = np.nanvar(sf_list_3)
            var_list_4 = np.nanvar(sf_list_4)
            var_list_5 = np.nanvar(sf_list_5)
            var_list_6 = np.nanvar(sf_list_6)

            add_file_list = [file_list_2[0], mse_180s, ssim_180s, nonzerocount_list_1, avg_list_1, std_list_1, var_list_1, mse_150s, ssim_150s, nonzerocount_list_2, avg_list_2, std_list_2, var_list_2, mse_120s, ssim_120s, nonzerocount_list_3, avg_list_3, std_list_3, var_list_3, mse_90s, ssim_90s, nonzerocount_list_4, avg_list_4, std_list_4, var_list_4, mse_60s, ssim_60s, nonzerocount_list_5, avg_list_5, std_list_5, var_list_5, mse_30s, ssim_30s, nonzerocount_list_6, avg_list_6, std_list_6, var_list_6]
            
            wr.writerow(add_file_list)

    f.close()

