#!/usr/bin/env python

import glob
import threading
import numpy as np
import multiprocessing
import cPickle as pickle
from netCDF4 import Dataset
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

def get_wrf_domain_lat_lon(file_name):
    flag=Dataset(file_name)
    lat=flag.variables['XLAT_M'][0,:,:]
    lon=flag.variables['XLONG_M'][0,:,:]
    flag.close()
    return np.min(lat),np.max(lat),np.min(lon),np.max(lon)

def get_cmip5_lat_lon_lev(file_name):
    name=glob.glob(file_name)[0]
    flag=Dataset(name)
    lat=flag.variables['lat'][:]
    lon=flag.variables['lon'][:]
    lev=flag.variables['plev'][:]
    if lev[0]<100000:
        lev*=10
    return lat,lon,lev

def get_cmip5_grid_index(crm_lllat,crm_urlat,crm_lllon,crm_urlon,cmip5_lat,cmip5_lon):
    dist_lat=np.sqrt(np.abs(cmip5_lat-crm_lllat))
    idx_y_lllat=np.unravel_index(dist_lat.argmin(),dist_lat.shape)[0]
    dist_lat=np.sqrt(np.abs(cmip5_lat-crm_urlat))
    idx_y_urlat=np.unravel_index(dist_lat.argmin(),dist_lat.shape)[0]
    dist_lon=np.sqrt(np.abs(cmip5_lon-crm_lllon))
    idx_x_lllon=np.unravel_index(dist_lon.argmin(),dist_lon.shape)[0]
    dist_lon=np.sqrt(np.abs(cmip5_lon-crm_urlon))
    idx_x_urlon=np.unravel_index(dist_lon.argmin(),dist_lon.shape)[0]
    return idx_y_lllat,idx_y_urlat,idx_x_lllon,idx_x_urlon

def get_cmip5_data_in_crm_grid(data,idx_y_lllat,idx_y_urlat,idx_x_lllon,idx_x_urlon):
    try:
        return data[:,min(idx_y_lllat,idx_y_urlat)-2:max(idx_y_lllat,idx_y_urlat)+2,\
                    min(idx_x_lllon,idx_x_urlon)-2:max(idx_x_lllon,idx_x_urlon)+2],\
               cmip5_lat[min(idx_y_lllat,idx_y_urlat)-2:max(idx_y_lllat,idx_y_urlat)+2],\
               cmip5_lon[min(idx_x_lllon,idx_x_urlon)-2:max(idx_x_lllon,idx_x_urlon)+2]
    except:
        return data[min(idx_y_lllat,idx_y_urlat)-2:max(idx_y_lllat,idx_y_urlat)+2,\
                    min(idx_x_lllon,idx_x_urlon)-2:max(idx_x_lllon,idx_x_urlon)+2],\
               cmip5_lat[min(idx_y_lllat,idx_y_urlat)-2:max(idx_y_lllat,idx_y_urlat)+2],\
               cmip5_lon[min(idx_x_lllon,idx_x_urlon)-2:max(idx_x_lllon,idx_x_urlon)+2]
    else:
        pass

def interp_thread(data_levels,interp_levels,data,temp,i,j,threadlock):
    f=interp1d(data_levels,data[:,i,j],kind='linear',fill_value='extrapolate')
    temp[:,i,j]=f(interp_levels)
    threadlock.release()

def process(data,rslt,data_levels,interp_levels,pid):
    shapes=np.shape(data)
    ny=shapes[1]
    nx=shapes[2]
    nz=np.shape(interp_levels)[0]
    temp=np.zeros([nz,ny,nx])
    threadlock=threading.BoundedSemaphore(128)
    for i in np.arange(0,ny,1):
        for j in np.arange(0,nx,1):
            threadlock.acquire()
            #f=interp1d(data_levels,data[:,i,j],kind='linear',fill_value='extrapolate')
            #temp[:,i,j]=f(interp_levels)
            t=threading.Thread(target=interp_thread,args=(data_levels,interp_levels,data,temp,i,j,threadlock))
            t.start()
            t.join()
    rslt[pid]=temp

"""test!!!test"""
model_name=raw_input()
month=raw_input()

interp_lev=np.array([1000,975,950,925,900,875,850,825,800,775,\
            750,700,650,600,550,500,450,400,350,300,\
            250,225,200,175,150,125,100,70,50,30,\
            20,10,7,5,3,2,1])*100

crm_lllat,crm_urlat,crm_lllon,crm_urlon=get_wrf_domain_lat_lon('grid.nc')
cmip5_lat,cmip5_lon,cmip5_lev=get_cmip5_lat_lon_lev('/data2/jptang/CMIP5-PGW/'+model_name+'/rcp85/hus_Amon_*_rcp85_*.nc')
idx_y_lllat,idx_y_urlat,idx_x_lllon,idx_x_urlon=get_cmip5_grid_index(crm_lllat,crm_urlat,crm_lllon,crm_urlon,cmip5_lat,cmip5_lon)

rslts={}
for cell_var in ['hus','ta','ua','va','zg','ps','psl']:
    data=pickle.load(open('/data2/jptang/PGW/monthly_mean/fcst/monthly_mean_'+model_name+'_'+month+'.pickle'))[cell_var]
    data_inuse,lat_inuse,lon_inuse=get_cmip5_data_in_crm_grid(data,idx_y_lllat,idx_y_urlat,idx_x_lllon,idx_x_urlon)
    rslts['lat']=lat_inuse
    rslts['lon']=lon_inuse
    if cell_var in ['ps','psl']:
        rslts[cell_var]=data_inuse
        #del data,data_inuse
    else:
        nz,ny,nx=np.shape(data_inuse)
        nz=np.shape(interp_lev)[0]
        cpu_count=multiprocessing.cpu_count()
        if cpu_count>max(ny,nx):
            cpu_count=max(ny,nx)
        if ny>nx:
            axis_type=1
        else:
            axis_type=2
        data_inuse_chunks=np.array_split(data_inuse,cpu_count,axis=axis_type)
        #del data,data_inuse

        manager=multiprocessing.Manager()
        trans=manager.dict()
        pool=multiprocessing.Pool(cpu_count)

        for idx,cell_data in enumerate(data_inuse_chunks):
            pool.apply_async(process,(cell_data,trans,cmip5_lev,interp_lev,idx))
        pool.close()
        pool.join()
        #print [np.shape(trans[cell]) for cell in sorted(trans.keys())]
        rslt=np.zeros([nz,ny,nx])
        if ny>nx:
            idx_sum=0
            for cell_key in sorted(trans.keys()):
                cell_nz,cell_ny,cell_nx=np.shape(trans[cell_key])
                #print model_name,cell_nz,cell_ny,cell_nx
                for i in np.arange(0,cell_ny,1):
                    for j in np.arange(0,cell_nx,1):
                        rslt[:,idx_sum+int(i),int(j)]=trans[cell_key][:,int(i),int(j)]
                idx_sum+=cell_ny
        else:
            idx_sum=0
            for cell_key in sorted(trans.keys()):
                cell_nz,cell_ny,cell_nx=np.shape(trans[cell_key])
                #print model_name,cell_nz,cell_ny,cell_nx
                for i in np.arange(0,cell_nx,1):
                    for j in np.arange(0,cell_ny,1):
                        try:
                            rslt[:,int(j),idx_sum+int(i)]=trans[cell_key][:,int(j),int(i)]
                        except:
                            print model_name,idx_sum+int(i),i,j
                idx_sum+=cell_nx
        rslts[cell_var]=rslt
    del data,data_inuse
flag_save=open('vertical_interp/fcst/CRM_domain_vertical_interp_'+model_name+'_'+month+'.pickle','wb')
pickle.dump(rslts,flag_save)
flag_save.close()
#plt.imshow(rslts['ta'][10,:,:])
#plt.colorbar()
#plt.show()

