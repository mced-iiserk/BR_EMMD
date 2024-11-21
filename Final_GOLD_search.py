#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import GMModel as gmm
from matplotlib import cm
import sys
from scipy.cluster.vq import kmeans2
from reWeight import *
import itertools
import matplotlib as mpl
import math
from tqdm import tqdm as tq


# In[2]:


def find_minima(file):
    
    x, y, z = np.load(file).T
    
    new_dim = int(np.sqrt(len(x)))
    X1 = x.reshape(new_dim,new_dim)
    Y1 = y.reshape(new_dim,new_dim)
    Z1 = z.reshape(new_dim,new_dim)
    
    minima = int(input("Enter total minima: "))
    upper_min = int(input("Upper cutoff of minima in kcal/mol:  "))
    extend = []
    positions = []
    #min_val   = np.min(arr)
    sort_arr   = np.sort(np.ndarray.flatten(Z1))
    min_subset = sort_arr[list(map(lambda x: x<upper_min, sort_arr))]
    
    for i in range (minima):
        xmin = float(input("Enter xmin value: "))
        xmax = float(input("Enter xmax value: "))
        ymin = float(input("Enter ymin value: "))
        ymax = float(input("Enter ymax value: "))
        extend.append([xmin,xmax,ymin,ymax])
        
    #print(extend)
    
    l=0
    while (l < len(min_subset)):
        j,k = np.where(np.isclose(Z1,min_subset[l]))
        positions.append([j[0],k[0],min_subset[l]])
        l=l+1
    #print(positions[0],positions[0][0],positions[0][1],positions[0][2])
    #selection of minima
    minima_dict = {}
    min_sub_subset_dict = {}
    for k in range(minima):
        min_sub_subset = []
        for j in range(len(min_subset)):
            if (extend[k][0]<=X1[positions[j][0],positions[j][1]]<=extend[k][1] and extend[k][2]<=Y1[positions[j][0],positions[j][1]]<=extend[k][3]):
                min_sub_subset.append(positions[j])
        min_sub_subset_dict["minima_%s" %(k)] = min_sub_subset
        min_ = min_sub_subset_dict["minima_%s" %(k)][np.argmin(np.min(min_sub_subset_dict["minima_%s" %(k)][:][2]))]
        minima_dict["minima_%s" %(k)] = [X1[min_[0],min_[1]],Y1[min_[0],min_[1]],min_[2]]
    
    return np.savez("min_sub_subset_dict.npz",**min_sub_subset_dict),np.savez("minima_dict.npz",**minima_dict)


# In[4]:


def angle_range(multiple,theta,points):
    range_ = np.linspace(theta - multiple*math.pi,theta + multiple*math.pi,points)
    return range_

def find_equation(file1,file2):
    minima_info = np.load(file1)
    x_, y_, z_ = np.load(file2).T
    
    new_dim = int(np.sqrt(len(x_)))
    X1 = x_.reshape(new_dim,new_dim)
    Y1 = y_.reshape(new_dim,new_dim)
    Z1 = z_.reshape(new_dim,new_dim)
    
    start  = input("enter starting minima (%s): " %(minima_info.files))
    end    = input("enter end minima (%s): " %(minima_info.files))
    steps  = 100  #int(input("enter steps to reach other minima: "))
    radius = float(input("enter radius cutoff: "))
    
    
    x2_ctr = minima_info[end][0]
    y2_ctr = minima_info[end][1]
    
    x1_ctr = minima_info[start][0]
    y1_ctr = minima_info[start][1]
    
    pts = 60 #int(input("enter intermediate points on circluar arc: "))
    outfile = f"min_{start[-1]}_{end[-1]}"
    path = [[x1_ctr,y1_ctr]]
    
    inFile = "memb_CV_dist.npy" #input("Enter AE_latent space output file : ")  
    MIX = np.load(inFile)
    # Train the gmm
    fitM = gmm.LearnFEL(MIX)
    fitM.trigger()
    mult  = 1
    dict_ = {}
    for i in tqdm(range(steps)):
        slope  = (y2_ctr - y1_ctr)/(x2_ctr - x1_ctr)
        x1 = (math.sqrt((radius**2)/(1+slope**2)) + x1_ctr)
        x2 = (x1_ctr - math.sqrt((radius**2)/(1+slope**2)))
        y1 = slope*(x1-x1_ctr) + y1_ctr
        y2 = slope*(x2-x1_ctr) + y1_ctr
        
        itrsec1 = np.array([x1,y1])
        itrsec2 = np.array([x2,y2])
        
        ctr2 = np.array([x2_ctr, y2_ctr])
        
        if np.linalg.norm(ctr2-itrsec1) < np.linalg.norm(ctr2-itrsec2):
            x=x1
            y=y1
        else:
            x=x2
            y=y2

        thet  = math.atan(slope)
        theta_range = angle_range(mult,thet,pts)

        x_arr = np.array(list(map(lambda z: (radius*math.cos(z) + x1_ctr), theta_range))) ## rcos(theta) = (X - center)
        y_arr = np.array(list(map(lambda z: (radius*math.sin(z) + y1_ctr), theta_range))) ## rcos(theta) = (X - center)

        meshX = np.array([x_arr,y_arr]).T

        PX = fitM.getPosterior(meshX)
        
        idxmax = PX.argmax()
        
        
        x1_ctr = 0.05*math.cos(theta_range[idxmax]) + x1_ctr
        y1_ctr = 0.05*math.sin(theta_range[idxmax]) + y1_ctr

        ctr1 = np.array([x1_ctr, y1_ctr])
        last = np.array(list(path[-1]))
        
        if np.linalg.norm(ctr2-ctr1) <= radius:
            print("Minimun energy path is ready")
            path.append([x1_ctr,y1_ctr])
            path.append([x2_ctr,y2_ctr])
            break
        elif np.linalg.norm(ctr1-last) <= radius*1e-1:
            print("Stuck at a point, changing algorihtm")
            mult = float(input("Enter multiple of pi: "))
            last = np.array(list(path[-2]))
        else:
            path.append([x1_ctr,y1_ctr])
            mult = 0.33
    return np.save(outfile+".npy",path)

def path_optimizer(file,rad=0.03,pts=20,gen_points=True,numiter=1):
    raw_path = np.load(file)
    int_corr = raw_path[1:-1,:].astype(dtype=np.float32)
    MIX  = np.load("memb_CV_dist.npy")
    fitM = gmm.LearnFEL(MIX)
    fitM.trigger()
    Prob = []
    if (gen_points==True):
        numiter=1
        for k in tq(range(0,numiter)):
            mid_pt_x = np.array((raw_path[1:,0] - raw_path[0:-1,0])/2 + raw_path[0:-1,0]).reshape(-1,1)
            mid_pt_y = np.array((raw_path[1:,1] - raw_path[0:-1,1])/2 + raw_path[0:-1,1]).reshape(-1,1)
            mid_pts  = np.append(mid_pt_x,mid_pt_y,axis=1).reshape(-1,2).astype(dtype=np.float32)
            new_corr = np.zeros(((mid_pts.shape[0]+int_corr.shape[0]),2))
            new_corr[np.arange(0,len(new_corr[:,0]),2),:] = mid_pts
            new_corr[np.arange(1,len(new_corr[:,0]),2),:] = int_corr
            idx = (0,new_corr.shape[0])
            new_corr = np.insert(new_corr,idx,[[raw_path[0,0],raw_path[0,1]],[raw_path[-1,0],raw_path[-1,1]]],axis=0)
            new_path = np.array([raw_path[0,0],raw_path[0,1]])
            #path_dict = ppd(new_corr,rad=rad,pts=pts)
            for i in tq(range(len(new_corr[1:-1,0]))):
                CV1_range = [new_corr[i,0]-rad,new_corr[i,0]+rad]
                CV2_range = [new_corr[i,1]-rad,new_corr[i,1]+rad]
                XX=np.linspace(*CV1_range,pts)
                YY=np.linspace(*CV2_range,pts)
                meshX = np.array(np.meshgrid(XX,YY)).T
                meshX = meshX.reshape(meshX.shape[0]**meshX.shape[-1],meshX.shape[-1])
                PX = fitM.getPosterior(meshX)
                PX = PX.reshape(pts,pts)
                j,k = np.where(np.isclose(PX,np.max(PX)))
                Prob.append(np.max(PX))
                x1 = XX[j[0]]
                y1 = YY[k[0]]
                new_path = np.append(new_path,np.array([x1,y1]),axis=0)
            new_path = np.append(new_path,np.array([raw_path[-1,0],raw_path[-1,1]]),axis=0)
            kB = 8.314/(4.2*1000)       # Boltzmann constant in unit kcal/mol/K
            T = 303.15                  # Temperature in Kelvin (K)
            kT = kB*T
            FE = -kT*np.log(np.array(Prob).reshape(-1,1))
            raw_path = new_path.reshape(-1,2)
    else:
        numiter=numiter
        for k in tq(range(0,numiter)):
            new_path = np.array([raw_path[0,0],raw_path[0,1]])
            for i in tq(range(len(int_corr[:,0]))):
                CV1_range = [int_corr[i,0]-rad,int_corr[i,0]+rad]
                CV2_range = [int_corr[i,1]-rad,int_corr[i,1]+rad]
                XX=np.linspace(*CV1_range,pts)
                YY=np.linspace(*CV2_range,pts)
                meshX = np.array(np.meshgrid(XX,YY)).T
                meshX = meshX.reshape(meshX.shape[0]**meshX.shape[-1],meshX.shape[-1])
                PX = fitM.getPosterior(meshX)
                PX = PX.reshape(pts,pts)
                j,k = np.where(np.isclose(PX,np.max(PX)))
                Prob.append(np.max(PX))
                x1 = XX[j[0]]
                y1 = YY[k[0]]
                new_path = np.append(new_path,np.array([x1,y1]),axis=0)
            new_path = np.append(new_path,np.array([raw_path[-1,0],raw_path[-1,1]]),axis=0)
            kB = 8.314/(4.2*1000)       # Boltzmann constant in unit kcal/mol/K
            T = 303.15                  # Temperature in Kelvin (K)
            kT = kB*T
            FE = -kT*np.log(np.array(Prob).reshape(-1,1))
            raw_path = new_path.reshape(-1,2)
    return np.save(file[:-4]+"_opt.npy",raw_path.reshape(-1,2)), np.save(file[:-4]+"_MFEP.npy",FE)

def gen_eqpath(file,rad=0.03):
    path = np.load(file).reshape(-1,2)
    dist = np.linalg.norm(path[1:,:] - path[0:-1,:],axis=1)
    mask = dist >= 0.03
    idx  =  np.arange(0,len(dist))[mask]
    gpath = path[0:idx[0],:].reshape(-1,2)
    for i in range(0,len(idx)):
        Y_vec = path[idx[i]+1][1] - path[idx[i]][1]
        X_vec = path[idx[i]+1][0] - path[idx[i]][0]
        #slope = Y_vec/X_vec
        #C     = Y_vec - (slope*X_vec)
        num = max(int((path[idx[i]+1][0]-path[idx[i]][0])/np.min(dist)),int((path[idx[i]+1][1]-path[idx[i]][1])/np.min(dist)))
        print(num)
        new_X = np.linspace(path[idx[i]][0],path[idx[i]+1][0],num)
        new_Y = np.linspace(path[idx[i]][1],path[idx[i]+1][1],num)
        #new_Y = slope*new_X #+ C
        ms_line = np.append(new_X.reshape(-1,1),new_Y.reshape(-1,1),axis=1)
        gpath = np.append(gpath,ms_line,axis=0)
        if idx[i] != idx[-1]:
            gpath = np.append(gpath,path[idx[i]:idx[i+1],:],axis=0)
        else:
            continue
    gpath = np.append(gpath,path[idx[-1]:,:].reshape(-1,2),axis=0)
    return np.save(file[:-4]+"_eqpath.npy",gpath)

