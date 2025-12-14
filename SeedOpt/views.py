from django.shortcuts import render

import pydicom as pyd
from pydicom.dataset import Dataset
from pprint import pprint
import base64
from io import BytesIO
import numpy as np
from django.contrib.staticfiles import finders
from PIL import Image
import os
import cupy as cp
import random

# Create your views here.

seedPos = []
def getSeedPos(file):
    seedPos = []
    file = finders.find(os.path.join("PL001.dcm"))
    plan = pyd.dcmread(file)
    for app in enumerate(plan.ApplicationSetupSequence):
        #print(f"Air Kerma   : {app.ApplicationSetupType}")
        for channel in app[1].ChannelSequence:
           # print(channel)
           # print("channel printed")
            #print(f"Number of Control Points   : {channel.NumberOfControlPoints}")
            for cop in enumerate(channel.BrachyControlPointSequence):  # every other is duplicate
                #print(cop[1])
                #print("print cp")
                if cop[1].ControlPointIndex == 0:
                    pos = cop[1].ControlPoint3DPosition
                
                    #print(f"control point          :{pos}")
                    seedPos.append(pos)
    return seedPos

def showPlan(request):
    seedPos = []
    file = finders.find(os.path.join("PL001.dcm"))
    plan = pyd.dcmread(file)
    print(f"Patient Name       : {plan.PatientName}")
    print(f"Patient ID         : {plan.PatientID}")
    print(f"Plan Name          : {plan.RTPlanLabel}")


    for src in plan.SourceSequence:
        #print(f"Source Type             : {src.SourceType}")                     # usually "POINT" or "LINE"
        
                      # e.g. I-125 OncoSeed, Pd-103, etc.
        print(f"Source Isotope          : {src.SourceIsotopeName}")

    
    seedPos  = getSeedPos(file)
    #print(f"seed positions : {seedPos}")
    seedPos = cp.array(seedPos)
    N = len(seedPos)
    xVals = seedPos[:,0]
    print(f"seed positions :{seedPos}")
    print(f"max X val :{cp.max(xVals)}")
    print(f"min X val :{cp.min(xVals)}")
    directions = [[0.0, 0.0, 1.0] for _ in range(N)]

    spacing = 1
    spacing_z = -5
    dfile = finders.find(os.path.join("DO001.dcm"))
    ds = pyd.dcmread(dfile)
    
    dose_dicom = ds.pixel_array.astype(cp.float32)
    dy, dx = float(ds.PixelSpacing[0]), float(ds.PixelSpacing[1])  # dy = row (y-dir), dx = column (x-dir)

# Starting position (corner of first voxel)
    x0, y0, z0 = float(ds.ImagePositionPatient[0]), float(ds.ImagePositionPatient[1]), float(ds.ImagePositionPatient[2])
    z_positions = cp.array(ds.GridFrameOffsetVector, dtype=cp.float32)

    nx, ny, nz = int(ds.Columns), int(ds.Rows), len(ds.GridFrameOffsetVector)
    gx = cp.arange(ds.Columns) + x0
    gy = cp.arange(ds.Rows) + y0    # Posterior → Anterior
    gz = z_positions
    #X, Y, Z = cp.meshgrid(grid_x, grid_y, grid_z, indexing='xy')
    #points = cp.stack((X.ravel(), Y.ravel(), Z.ravel()), axis=1)
   
    print(f"z grid array: {gz}")
    dose = calcDose(seedPos, gx, gy, gz, output_dose_rate=False)
    dfile = finders.find(os.path.join("DO001.dcm"))
    plan_dose = getDose(dfile)
    getDoseErr(dose, plan_dose)

    #print(f"final calced dose: {dose}")
    #calcDoseCPU(file, dfile)
    return render(request, 'home.html', {})

def calcDose(seed_pos, grid_x, grid_y, grid_z, output_dose_rate=True):
    Lambda = 0.965
    L = 0.45  # cm
    gL_r = cp.array([0.1, 0.15, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0,
                     4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    gL_val = cp.array([0.696, 0.853, 0.982, 1.048, 1.036, 1.00, 0.912, 0.819, 0.636,
                       0.499, 0.367, 0.272, 0.20, 0.149, 0.110, 0.0809])
    
    ani_r = cp.array([0.5, 1.0, 2.0, 3.0, 4.0, 5.0])
    
    ani_val  = cp.array([0.973, 0.944, 0.941, 0.942, 0.943, 0.944]) 
    
    X, Y, Z = cp.meshgrid(grid_x, grid_y, grid_z, indexing='xy')
    points = cp.stack((X.ravel(), Y.ravel(), Z.ravel()), axis=1)
    print(f"points shape: {points.shape}")
    print(f"X grid size:  {len(X)}")


    if isinstance(seed_pos, list):
        seeds = cp.asarray(np.array(seed_pos), dtype=cp.float32)
    else:
        seeds = cp.asarray(seed_pos, dtype=cp.float32)
    
    pos = seeds[:, 0:3]  # (Nseeds, 3)
    #dir_vec = seeds[:, 3:6]  # (Nseeds, 3)

    Nseeds, Npoints = pos.shape[0], points.shape[0]

    points_exp = points[None, :, :]  # (1, Npoints, 3)
    pos_exp = pos[:, None, :]        # (Nseeds, 1, 3)
   
    print(f"number of point: {Npoints}")
    
    #print(f"pos exp : {pos_exp}")
    #print(f"dose points:  {points_exp}")

    vec = points_exp - pos_exp
    #print(f"vec :  {vec}")
    r= cp.sqrt(cp.sum(vec**2, axis=2))
    #print(f"radius :  {r}")
    print(f"radius size :  {r.shape}")
    r = r/10
    r = cp.maximum(r, 0.2)
    
    #print(f"r  array:  {r}")
    def interp_aniso(rs):
        out = cp.zeros_like(rs)
        for i in range(1, len(ani_r)):
            mask = (rs >= ani_r[i-1]) & (rs < ani_r[i])
            t = (rs - ani_r[i-1]) / (ani_r[i] - ani_r[i-1])
            out = cp.where(mask, ani_val[i-1] + t * (ani_val[i] - ani_val[i-1]), out)
        out = cp.where(rs >= 5.0, ani_val[-1], out)
        out = cp.where(rs <= 0.5, ani_val[0], out)
        return out


    def interp_gL(rs):
        # rs: (Nseeds, Npoints) -> (Nseeds, Npoints)
        out = cp.zeros_like(rs)
        for i in range(1, len(gL_r)):
            mask = (rs >= gL_r[i-1]) & (rs < gL_r[i])
            t = (rs - gL_r[i-1]) / (gL_r[i] - gL_r[i-1])
            out = cp.where(mask, gL_val[i-1] + t * (gL_val[i] - gL_val[i-1]), out)
        # Beyond 10 cm: 1/r² tail
        out = cp.where(rs >= 10.0, gL_val[-1], out)
        return out
    gL = interp_gL(r)
    aniso = interp_aniso(r)

    print(f"gL  array:  {cp.max(gL)}")
    # 1D anisotropy φ_an(r): simple fit, broadcasts
    

    dose = cp.sum(0.427 * Lambda * 1/r**2 * gL * aniso * 2057, axis=0)
    dose = dose/100
    #print(f"dose : {dose}")
    #print(f"dose points : {dose.shape}")

    dose3d = dose.reshape((len(grid_x), len(grid_y), len(grid_z)))

    #print(f"dose3d : {dose3d}")
    dose3d = np.transpose(dose3d,(2,0,1))

    dose3d *= 1.0

    #print(f"dose3d processed: {dose3d}")

    print(f"max calc dose: {cp.max(dose3d):.2f} Gy")
    print(f"min calc dose: {cp.min(dose3d):.2f} Gy")
    print(f"mean calc dose: {cp.mean(dose3d):.2f} Gy")
    print(f"num points: {cp.shape(dose3d)[0]*cp.shape(dose3d)[1]*cp.shape(dose3d)[2]}")


    return dose3d.get()

def showUS(request):

    file2 = finders.find(os.path.join("Images","US001.dcm"))
    ds =  pyd.dcmread(file2)
    pix_data = ds.pixel_array
    total_frames = 1
    is_multiframe = False
    pix_data.dtype 
    total_frames = 1
    is_multiframe = False
    print(f"data type:  {pix_data.dtype}") 
    img = Image.fromarray(pix_data)

def getDose(file):
    ds = pyd.dcmread(file)
    dose_grid  = ds.pixel_array
    #print(dose_grid)
    print(dose_grid.shape)       # Usually (Frames, Rows, Columns) → e.g. (120, 256, 256)
    print(dose_grid.dtype)       # Usually uint32 or uint16
    print(f"Grid offset: {ds.GridFrameOffsetVector}")
    scaled_dose = dose_grid * float(ds.DoseGridScaling)
    dose_gy = scaled_dose.astype(np.float32)
    
    print(f"Max dose: {dose_gy.max():.2f} Gy")
    print(f"Min dose: {dose_gy.min():.2f} Gy")
    print(f"Mean dose: {dose_gy.mean():.2f} Gy")
    print(f"width: {ds.PixelSpacing[0] * ds.Columns:.2f} mm")
    print(f"height: {ds.PixelSpacing[1] * ds.Rows:.2f} mm")
    print(f"num dose points: {dose_gy.shape[0]*dose_gy.shape[1]*dose_gy.shape[2]}")
    return dose_gy
   
    
def getDoseErr(calc_dose, plan_dose):
    diff = (calc_dose - plan_dose)/plan_dose*100
    min_100 = np.sort(np.abs(diff).flatten() )[:5000]
    diff_sorted  = np.argsort(np.abs(diff).flatten())
    x, y, z = np.unravel_index(np.argmax(diff), diff.shape )
    print(f"max diff pos: {x} {y} {z}")    
    #print(f"dose_diff:  { diff}")
    '''for i in diff_sorted[50000:50100]:
         print(plan_dose.shape)
         print(calc_dose.shape)   
         z, y, x = np.unravel_index(i, diff.shape)
         print(f"diff pos: {x} {y} {z}") 
         print(f"diff val {diff[z][x][y]}")
         print(f"calc val {calc_dose[z][x][y]}")
         print(f"plan val {plan_dose[z][x][y]}")'''
    print(f"diff mean: {cp.mean(diff)}")

def calcDoseCPU(Pfile, Dfile): 
    seedPos = getSeedPos(Pfile)
    dose_grid =  getDose(Dfile)
    ds = pyd.dcmread(Dfile)
    dose_grid  = ds.pixel_array * ds.DoseGridScaling


    gL_r = np.array([0.1, 0.15, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0,
                     4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    gL_val = np.array([1.055, 1.078, 1.082, 1.071, 1.042, 1.00, 0.908, 0.814, 0.632,
                       0.496, 0.364, 0.270, 0.199, 0.148, 0.109, 0.0803])
    
    ani_r = np.array([0.5, 1.0, 2.0, 3.0, 4.0, 5.0])
    
    ani_val  = np.array([0.973, 0.944, 0.941, 0.942, 0.943, 0.944]) 
    print(f"number of seeds:  {seedPos}")
    def interp_aniso(rs):
        val =1
        for i in range(1, len(ani_r)):
            if(rs >= ani_r[i-1]) & (rs < ani_r[i]):
                t = (rs - ani_r[i-1]) / (ani_r[i] - ani_r[i-1])
                val =  ani_val[i-1] + t * (ani_val[i] - ani_val[i-1])
        if rs >= 5.0: 
            val = ani_val[-1]
        
        if rs <= 0.5:
            val =  ani_val[0]
            
        return val

    def interp_gL(rs):
        # rs: (Nseeds, Npoints) -> (Nseeds, Npoints)
        val = 1
        for i in range(1, len(gL_r)):
            if (rs >= gL_r[i-1]) & (rs < gL_r[i]):
                t = (rs - gL_r[i-1]) / (gL_r[i] - gL_r[i-1])
                val = gL_val[i-1] + t * (gL_val[i] - gL_val[i-1])
        # Beyond 10 cm: 1/r² tail
        if rs >= 10.0: 
            val = gL_val[-1]
        return val
    Lambda = 0.965
    
    doseCalc = 0
    diffList = []
    calcList = []
    for i in range(100000):
        
        
        x = random.randint(1,119)
        y = random.randint(1,119)
        z = random.randint(0,6)
        x_pos = ds.ImagePositionPatient[0]+x
        y_pos = ds.ImagePositionPatient[1]+y
        z_pos = ds.GridFrameOffsetVector[z]

        dose_val = dose_grid[z][y][x]
        doseCalc = 0
        if dose_val > 0.5:
            for seed in seedPos:
                r =  np.sqrt((seed[0] - x_pos)**2 + (seed[1] - y_pos)**2 + (seed[2] - z_pos)**2)/10
                r = np.maximum(r, 0.15)
                gL = interp_gL(r)
                aniso = interp_aniso(r)
                doseCalc += 0.427 * Lambda * (1/r**2) * gL * aniso * 2057
            doseCalc = doseCalc/100
            print(f"calc dose value: {doseCalc} Gy")
            print(f"plan dose value: {dose_val} Gy")
        
            doseDiff = (doseCalc - dose_val)/dose_val *100
            print(f"dose diff : {doseDiff} %")
            diffList.append(doseDiff)
            calcList.append(doseCalc)
    diffArray = np.array(diffList)
    calcArray = np.array(calcList)
    print(f"mean diff: {np.mean(np.abs(diffArray))}")
    print(f"max diff: {np.max(diffArray)}")
    print(f"min diff: {np.min(diffArray)}")
    print(f"number of seeds:  {len(seedPos)}")
    print(f"mean calc dose: {np.mean(calcArray)}")
    print(f"mean plan dose: {np.mean(dose_grid)}")

#def createPoints(x_grid, y_grid, z_grid):
    
    #for z in z_grid:





         




