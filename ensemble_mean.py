#!/usr/bin/env python

import threading
import numpy as np
import multiprocessing
import cPickle as pickle
import matplotlib.pyplot as plt
from ctypes import *

def read_data_thread(path,var,test,month,rslt,threadlock):
    rslt[test]=pickle.load(open(path+'/'+var+'/'+var+'_'+test+'_'+month+'.pickle'))
    threadlock.release()

def read_data(path,var,tests,month,rslt):
    threadlock=multiprocessing.BoundedSemaphore(4)
    for test in tests:
        threadlock.acquire()
        t=threading.Thread(target=read_data_thread,args=(path,var,test,month,rslt,threadlock))
        t.start()
        t.join()

def read_ensemble_members(path,month,var):
    tests=['ACCESS1_0','BCC-CSM1-1','CanESM2','CCSM4',\
            'CMCC-CM','CNRM-CM5','CSIRO-Mk3_6','FGOALS-S2',\
            'GFDL-CM3','GFDL-ESM2M','HadGEM2-ES','inmcm4',\
            'IPSL-CM5A-LR','MRI-ESM','NorESM1-M']
    cpu_count=multiprocessing.cpu_count()
    if len(tests)<cpu_count:
        cpu_count=len(tests)
    test_chunks=np.array_split(np.asarray(tests),cpu_count,axis=0)
    manager=multiprocessing.Manager()
    rslt=manager.dict()
    pool=multiprocessing.Pool(cpu_count)

    for cell in test_chunks:
        pool.apply_async(read_data,(path,var,cell,month,rslt))
    pool.close()
    pool.join()
    keys=rslt.keys()
    nz,ny,nx=np.shape(rslt[keys[0]])
    return rslt,nz,ny,nx

def calculate_ensemble_mean(names,level,data,rslt):
    tests1=['ACCESS1_0','BCC-CSM1-1','CanESM2','CCSM4',\
            'CMCC-CM','CNRM-CM5','CSIRO-Mk3_6','FGOALS-S2',\
            'GFDL-CM3','GFDL-ESM2M','HadGEM2-ES','inmcm4',\
            'IPSL-CM5A-LR','MRI-ESM','NorESM1-M']
    tests2=['BCC-CSM1-1','CanESM2',\
            'CMCC-CM','CNRM-CM5','FGOALS-S2',\
            'inmcm4','NorESM1-M']
    if level<=14:
        tests_use=tests2
    else:
        tests_use=tests1
    trans=[]
    for cell in tests_use:
        trans+=[data[names.index(cell),int(level),:,:]]
    rslt[int(level),:,:]=np.mean(np.asarray(trans),axis=0)

def process(keys,levels,data,rslt,processlock):
    names=keys[:]
    for level in levels:
        calculate_ensemble_mean(names,level,data,rslt)
    processlock.release()

"""test!!!test"""
var=raw_input()
month=raw_input()
data,nz,ny,nx=read_ensemble_members('/data2/jptang/PGW/horizontal_interp/fcst/',month,var)
val=[]
key_=[]
for cell in data.keys():
    val+=[data[cell]]
    key_+=[cell]

keys=data.keys()
shared_array_input_base=multiprocessing.Array(c_double,len(keys)*nz*ny*nx)
shared_array_input=np.ctypeslib.as_array(shared_array_input_base.get_obj())
shared_array_input=shared_array_input.reshape(len(keys),nz,ny,nx)
shared_array_input[:,:,:,:]=np.array(val)[:,:,:,:]
del val

cpu_count=multiprocessing.cpu_count()
if nz<cpu_count:
    cpu_count=nz
level_chunks=np.array_split(np.arange(0,nz,1),cpu_count,axis=0)
shared_array_base=multiprocessing.Array(c_double,nz*ny*nx)
shared_array=np.ctypeslib.as_array(shared_array_base.get_obj())
shared_array=shared_array.reshape(nz,ny,nx)

processlock=multiprocessing.BoundedSemaphore(cpu_count)
for cell in level_chunks:
    processlock.acquire()
    p=multiprocessing.Process(target=process,args=(keys,cell,shared_array_input,shared_array,processlock))
    p.start()
    p.join()

flag_save=open('ens_mean_'+var+'_'+month+'.pickle','wb')
pickle.dump(shared_array,flag_save)
flag_save.close()
#print shared_array
#plt.imshow(shared_array[0,:,:])
#plt.show()
