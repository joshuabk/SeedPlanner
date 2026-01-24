from django.shortcuts import render

import pydicom as pyd
from pydicom.dataset import Dataset
from pprint import pprint
import base64
from io import BytesIO
import numpy as np
from django.contrib.staticfiles import finders
from pydicom.pixel_data_handlers.util import apply_voi_lut, convert_color_space
from PIL import Image, ImageDraw
import os
import cupy as cp
import random

import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.ndimage import zoom  # for simple resampling

# Create your views here.

def showUS(request):

    folder = finders.find(os.path.join("Images","Richard_Gough"))
    path = finders.find(os.path.join('files', 'DO001RG.dcm'))

    dose_ds = pyd.dcmread(path)

    if not folder or not os.path.isdir(folder):
        return render(request, 'Images.html', {'error': 'Folder not found'})

    dicom_files = sorted([f for f in os.listdir(folder) if f.lower().endswith('.dcm')])

    all_images = []
    num_images = 0
    structs = getStructures()
    sliceNum =0
    seedPos = getSeedPos()
    for file_name in dicom_files:

        file_path = os.path.join(folder, file_name)
        ds = pyd.dcmread(file_path)


        ds =  pyd.dcmread(file_path)
        arr = ds.pixel_array
        total_frames = 1
        print(f"data type:  {arr.dtype}") 
        #img = Image.fromarray(pix_data)

        if hasattr(ds, 'PhotometricInterpretation'):
            if ds.PhotometricInterpretation in ['YBR_FULL', 'YBR_FULL_422']:
                arr = convert_color_space(arr, ds.PhotometricInterpretation, 'RGB')
            arr = apply_voi_lut(arr, ds, prefer_lut=True)

        if arr.dtype != np.uint8:
            arr_min, arr_max = arr.min(), arr.max()
            arr = np.uint8((arr - arr_min) / (arr_max - arr_min + 1e-8) * 255)

        if len(arr.shape) == 3:  # RGB
            img = Image.fromarray(arr, mode='RGB')
        else:  # Grayscale
            img = Image.fromarray(arr, mode='L')

        
        img_with_dose = overlay_dose_on_us(
        us_ds=ds,
        dose_ds=dose_ds,
        img_pil=img,
        slice =  sliceNum,
        alpha=0.3,           # semi-transparent
        colormap='jet'        # or 'hot', 'coolwarm', etc.
    )
        img_with_dose = img_with_dose.convert("RGB")
        drawSeeds(seedPos, img_with_dose, sliceNum) 
        if sliceNum < len(structs):
            drawPoly(structs[sliceNum], img_with_dose, color = 'red')
        buffer = BytesIO()
        img_with_dose.save(buffer, format='PNG')
        #img.save(buffer, format='PNG')
        img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        all_images.append((img_base64, file_name))
        
        
        sliceNum  += 1
    
    # Pass to template
    return render(request, 'Images.html', {'US_images': all_images, "num_images": len(dicom_files)})

def getStructures():
    path = finders.find(os.path.join('files', 'SS001RG.dcm'))

    rt_ds = pyd.dcmread(path)
    contour_slices = []
    #for set in rt_ds.StructureSetROISequence:
        
    for cont in rt_ds.ROIContourSequence:
            color = [float(num) for num in cont.ROIDisplayColor]
            for slice in cont.ContourSequence:
               
                slice_points = slice.ContourData

                points  = [[float(slice_points[i]), float(slice_points[i+1]), float(slice_points[i+2])] for i in range(0, len(slice_points), 3)]

                
                contour_slices.append(points)
    for slice in contour_slices:
        for point in points:
                print(point)

    return contour_slices

def drawPoly(contour_slice, img, color):
    Ppath = finders.find(os.path.join('files', 'DO001RG.dcm'))
    USfile = finders.find(os.path.join("Images","Richard_Gough", 'US001.dcm'))
    p_ds = pyd.dcmread(Ppath)
    Idcm = pyd.dcmread(USfile)
    origin = np.array(p_ds.ImagePositionPatient[:2])
    pix_spacing =  np.array(Idcm.PixelAspectRatio)
    X_delta = float(Idcm.PhysicalDeltaX)
    Y_delta = float(Idcm.PhysicalDeltaY)
    orientation = np.array(p_ds.ImageOrientationPatient)
    row_dir = orientation[3:]
    col_dir = orientation[:3]

    draw = ImageDraw.Draw(img)
    poly_pix = []
    for x, y, z in contour_slice:
        vec = np.array([ x,y, 0]) 
        pixel_x  = np.dot(vec, col_dir) / (X_delta*10)
        pixel_y = np.dot(vec, row_dir) / (Y_delta*10)
        poly_pix.append((pixel_x, pixel_y))
    draw.polygon(poly_pix, outline= color, width=3)




def overlay_dose_on_us(us_ds, dose_ds, img_pil, slice, alpha=0.0, colormap='jet'):
    """
    Overlay RTDOSE isodose/color wash on ultrasound PIL Image.
    
    Parameters:
        us_ds: pydicom Dataset of current ultrasound slice
        dose_ds: pydicom Dataset of RTDOSE
        img_pil: PIL Image of ultrasound (grayscale or RGB)
        alpha: transparency of dose overlay (0.0–1.0)
        colormap: matplotlib colormap (e.g., 'jet', 'hot', 'viridis')
    """
    # 1. Get dose grid (scaled physical dose in Gy)
    dose_array = dose_ds.pixel_array.astype(float) * dose_ds.DoseGridScaling
    dose_slice = dose_array[slice]
    #dose_slice = np.rot90(dose_slice, k=1)  # often need rotation/flip to match orientation
    dose_offsets = np.array(dose_ds.GridFrameOffsetVector)

    # 2. Dose grid geometry
    dose_origin = np.array(dose_ds.ImagePositionPatient)
    dose_spacing = np.array(dose_ds.PixelSpacing)
    dose_orientation = np.array(dose_ds.ImageOrientationPatient)

    # 3. Ultrasound geometry (reference for overlay)
    
    us_shape = np.array([us_ds.Columns, us_ds.Rows])
    # 4. Simple resampling: scale dose to match US size (approximate alignment)
    # Compute scaling factors (this is rough — for better use full affine transform)
    #print(f"dose slice shape {dose_slice.shape}" )
    #print(f"us x shape {us_shape[0]}" )
    X_delta = float(us_ds.PhysicalDeltaX)
    Y_delta = float(us_ds.PhysicalDeltaY)
    scale_x = us_shape[0]/dose_slice.shape[0]
    scale_y =  us_shape[1]/dose_slice.shape[1]
    #print("dose shape")
    # print(dose_array.shape)
    dose_resampled = zoom(dose_slice, (scale_y, scale_x), order=1)  # bilinear interp
    #print(f"fisrt dose resampled shape {dose_resampled.shape}")

    # Clip/resample to match US shape (crop or pad if needed)
    dose_resampled = dose_resampled[ :us_shape[1],  :us_shape[0]]

    # Normalize dose for colormap (e.g., 0–20 Gy → 0–1)
    print(f"dose resampled shape {dose_resampled.shape}")
    #print(f"dose resampled values {dose_resampled}")
    max = np.max(dose_resampled)
    print(f"max dose is {max}")
    dose_norm = np.clip(dose_resampled/100, 0, 1)  # adjust max dose as needed

    # 5. Create color overlay using colormap
    cmap = cm.get_cmap(colormap)
    dose_color = cmap(dose_norm)  # RGBA array
    #print(f"dose color shape {dose_color.shape}")
    mask = dose_norm > 0.05   # show only dose > 5%
    dose_color[..., 3] = np.where(mask, dose_color[..., 3] * alpha, 0)
    
    dose_color = (dose_color * 255).astype(np.uint8)  # to uint8
    #print(f"dose color shape 2 {dose_color.shape}")
    # 6. Blend with ultrasound image
    us_array = np.array(img_pil.convert('RGB'))
    overlay = Image.fromarray(dose_color, mode='RGBA')
    us_img = Image.fromarray(us_array)
    #print(f"overlay shape {overlay.size}")
    #print(f"image shape {us_img.size}")
    # Blend with alpha
    blended = Image.alpha_composite(us_img.convert('RGBA'), overlay)

    return blended.convert('RGB')  # back to RGB for saving


def getSeedPos():
    seedPos = []
    file = finders.find(os.path.join("files", "PL001RG.dcm"))
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

def drawSeeds(seed_pos, img, slice_num):
    USfile = finders.find(os.path.join("Images","Richard_Gough", 'US001.dcm'))
    Ppath = finders.find(os.path.join('files', 'DO001RG.dcm'))
    draw = ImageDraw.Draw(img)

    us_ds = pyd.dcmread(USfile)
    dose_ds = pyd.dcmread(Ppath)
    X_delta = float(us_ds.PhysicalDeltaX)
    Y_delta = float(us_ds.PhysicalDeltaY)
    
    orientation = np.array(dose_ds.ImageOrientationPatient)
    origin = np.array(dose_ds.ImagePositionPatient[:2])
    row_dir = orientation[3:]
    col_dir = orientation[:3]
    
    zpos = slice_num * -5
    for seed in seed_pos:
        if seed[2] == zpos:
            x , y  = seed[:2] 
            vec = np.array([ x,y, 0]) 
            usx = np.dot(vec, col_dir) / (X_delta*10)
            usy = np.dot(vec, row_dir) / (Y_delta*10)
            bbox = [usx-5, usy-5, usx+5, usy+5]

            draw.ellipse(bbox, fill="green", outline=(0,0,0), width=1)
            
    return img


    print(f"seed positions {seed_pos}")
    #for seed in seed_pos:
    #    if seed[2]
     


class StructureSet:
    def __init__(self):
        
        self.Structures = []
        self.Plan
        
  

class Structure:
    def __init__(self, color, slices, name):
        self.color  = color
        self.slices = []
        self.name = name
        self.Plan

class contourSlice:
     def __init__(self, zval, points):
        self.zVal = zval
        self.points = points

class ImageSet:
     def __init__(self):
         self.Images = []
         self.Plan

class USImage:
    def __init__(self, pixels, zval):
        self.pixelArray 
        self.zval
        self.contours















    


