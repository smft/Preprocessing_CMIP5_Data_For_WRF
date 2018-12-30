#!/usr/bin/env python

import threading
import numpy as np
import multiprocessing
import cPickle as pickle
import matplotlib.pyplot as plt
from netCDF4 import Dataset
from scipy.ndimage.filters import gaussian_filter

def calculate_idw_factor(crm_lat,crm_lon,crm_i,crm_j,cmip5_lat_2d,cmip5_lon_2d,rslt,threadlock):
    dist=np.sqrt((cmip5_lat_2d-crm_lat)**2+(cmip5_lon_2d-crm_lon)**2)
    nearest_four_dist=np.sort(dist,axis=None)[:9]
    for i,cell in enumerate(nearest_four_dist):
        idy,idx=np.where(dist==cell)
        rslt[i*3+0,crm_i,crm_j]=idy[0]
        rslt[i*3+1,crm_i,crm_j]=idx[0]
        rslt[i*3+2,crm_i,crm_j]=cell*100
    threadlock.release()

def process(idx_y,idx_x,crm_lat_2d,crm_lon_2d,cmip5_lat_1d,cmip5_lon_1d,rslt):
    proc_crm_lat_2d=crm_lat_2d[:,:]
    proc_crm_lon_2d=crm_lon_2d[:,:]
    ny,nx=np.shape(crm_lat_2d)
    trans=np.zeros([27,ny,nx])
    cmip5_lon_2d,cmip5_lat_2d=np.meshgrid(cmip5_lon_1d,cmip5_lat_1d)
    threadlock=threading.BoundedSemaphore(128)
    for cell_y in idx_y:
        for cell_x in idx_x:
            threadlock.acquire()
            #print type(cell_y),type(cell_x)
            t=threading.Thread(target=calculate_idw_factor,args=(proc_crm_lat_2d[cell_y,cell_x],\
                                                                proc_crm_lon_2d[cell_y,cell_x],\
                                                                cell_y,cell_x,cmip5_lat_2d,\
                                                                cmip5_lon_2d,trans,threadlock))
            t.start()
            t.join()
            #calculate_idw_factor(proc_crm_lat_2d[cell_y,cell_x],\
            #                    proc_crm_lon_2d[cell_y,cell_x],\
            #                    cell_y,cell_x,cmip5_lat_2d,\
            #                    cmip5_lon_2d,trans,0)
    rslt+=[trans]

def interp_one_point(factor,cmip5_data,crm_i,crm_j,rslt,threadlock):
    rslt[:,crm_i,crm_j]=((cmip5_data[:,int(factor[0]),int(factor[1])]/factor[2])+\
                        (cmip5_data[:,int(factor[3]),int(factor[4])]/factor[5])+\
                        (cmip5_data[:,int(factor[6]),int(factor[7])]/factor[8])+\
                        (cmip5_data[:,int(factor[9]),int(factor[10])]/factor[11])+\
                        (cmip5_data[:,int(factor[12]),int(factor[13])]/factor[14])+\
                        (cmip5_data[:,int(factor[15]),int(factor[16])]/factor[17])+\
                        (cmip5_data[:,int(factor[18]),int(factor[19])]/factor[20])+\
                        (cmip5_data[:,int(factor[21]),int(factor[22])]/factor[23])+\
                        (cmip5_data[:,int(factor[24]),int(factor[25])]/factor[26]))/\
                        ((1/factor[2])+(1/factor[5])+(1/factor[8])+(1/factor[11])+\
                        (1/factor[14])+(1/factor[17])+(1/factor[20])+(1/factor[23])+(1/factor[26]))
    #rslt[:,crm_i,crm_j]=(cmip5_data[:,int(factor[0]),int(factor[1])]+\
    #                     cmip5_data[:,int(factor[3]),int(factor[4])]+\
    #                     cmip5_data[:,int(factor[6]),int(factor[7])]+\
    #                     cmip5_data[:,int(factor[9]),int(factor[10])])/4
    threadlock.release()

def process_interp(idx_y,idx_x,cmip5_data,factor,rslt):
    temp_data=cmip5_data[:,:,:]
    nz=np.shape(temp_data)[0]
    temp_factor=factor[:,:,:]
    nz_old,ny,nx=np.shape(temp_factor)
    trans=np.zeros([nz,ny,nx])
    threadlock=threading.BoundedSemaphore(128)
    for cell_y in idx_y:
        for cell_x in idx_x:
            threadlock.acquire()
            t=threading.Thread(target=interp_one_point,args=(temp_factor[:,cell_y,cell_x],temp_data,cell_y,cell_x,trans,threadlock))
            t.start()
            t.join()
    rslt+=[trans]


def guass_flt(idxs,data,rslt):
    temp=data[:,:,:]
    for idx in idxs:
        rslt[int(idx)]=gaussian_filter(temp[int(idx),:,:],[12,12],mode='nearest')

"""test!!!test"""
var=raw_input()
test_name=raw_input()
month=raw_input()

if var=='ua':
    crm_lat=Dataset('grid.nc').variables['XLAT_U'][0,:,:]
    crm_lon=Dataset('grid.nc').variables['XLONG_U'][0,:,:]
elif var=='va':
    crm_lat=Dataset('grid.nc').variables['XLAT_V'][0,:,:]
    crm_lon=Dataset('grid.nc').variables['XLONG_V'][0,:,:]
else:
    crm_lat=Dataset('grid.nc').variables['XLAT_M'][0,:,:]
    crm_lon=Dataset('grid.nc').variables['XLONG_M'][0,:,:]

name_vertical_interp='/data2/jptang/PGW/vertical_interp/fcst/CRM_domain_vertical_interp_'+test_name+'_'+month+'.pickle'
cmip5_lat_1d=pickle.load(open(name_vertical_interp))['lat']
cmip5_lon_1d=pickle.load(open(name_vertical_interp))['lon']
cmip5_data=pickle.load(open(name_vertical_interp))[var]

cpu_count=multiprocessing.cpu_count()
ny,nx=np.shape(crm_lat)
idx_y=np.arange(0,ny,1)
idx_x=np.arange(0,nx,1)

if cpu_count>max(ny,nx):
    cpu_count=max(ny,nx)
# calculate interp factor
manager=multiprocessing.Manager()
rslt=manager.list()
pool=multiprocessing.Pool(cpu_count)

if ny>nx:
    idx_y_chunks=np.array_split(idx_y,cpu_count,axis=0)
    idx_x_chunks=idx_x
    for cell in idx_y_chunks:
        pool.apply_async(process,(cell,idx_x_chunks,crm_lat,crm_lon,cmip5_lat_1d,cmip5_lon_1d,rslt))
    pool.close()
    pool.join()
else:
    idx_y_chunks=idx_y
    idx_x_chunks=np.array_split(idx_x,cpu_count,axis=0)
    for cell in idx_x_chunks:
        pool.apply_async(process,(idx_y_chunks,cell,crm_lat,crm_lon,cmip5_lat_1d,cmip5_lon_1d,rslt))
    pool.close()
    pool.join()
factor=np.sum(np.asarray(rslt),axis=0)

# 9 points idw
manager=multiprocessing.Manager()
rslt=manager.list()
pool=multiprocessing.Pool(cpu_count)

if ny>nx:
    idx_y_chunks=np.array_split(idx_y,cpu_count,axis=0)
    idx_x_chunks=idx_x
    for cell in idx_y_chunks:
        pool.apply_async(process_interp,(cell,idx_x_chunks,cmip5_data,factor,rslt))
    pool.close()
    pool.join()
else:
    idx_y_chunks=idx_y
    idx_x_chunks=np.array_split(idx_x,cpu_count,axis=0)
    for cell in idx_x_chunks:
        pool.apply_async(process_interp,(idx_y_chunks,cell,cmip5_data,factor,rslt))
    pool.close()
    pool.join()
data_interp=np.sum(np.asarray(rslt),axis=0)

# gaussian filter
cpu_count=multiprocessing.cpu_count()
nz,ny,nx=np.shape(data_interp)
if nz<cpu_count:
    cpu_count=nz
data_interp_chunks=np.array_split(np.arange(0,nz,1),cpu_count,axis=0)
manager=multiprocessing.Manager()
rslt=manager.dict()
pool=multiprocessing.Pool(cpu_count)
for cell in data_interp_chunks:
    pool.apply_async(guass_flt,(cell,data_interp,rslt))
pool.close()
pool.join()
#print rslt

# save result
flag_save=open('/data2/jptang/PGW/horizontal_interp/fcst/'+var+'_'+test_name+'_'+month+'.pickle','wb')
pickle.dump(np.asarray([rslt[cell] for cell in sorted(rslt.keys())]),flag_save)
flag_save.close()

