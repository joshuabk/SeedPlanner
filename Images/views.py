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
import SimpleITK as sitk

import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.ndimage import zoom  # for simple resampling

from django.shortcuts import render
from typing import List, Optional
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST
import json




import time

# Create your views here.
numSlices = 0

planPath = finders.find(os.path.join('files', 'PL001RG.dcm'))
def showUS(request):

    folder_us = finders.find(os.path.join("Images","Richard_Gough"))
    dosePath = finders.find(os.path.join('files', 'DO001RG.dcm'))
    

    dose_ds = pyd.dcmread(dosePath)

    if not folder_us or not os.path.isdir(folder_us):
        return render(request, 'Images.html', {'error': 'Folder not found'})

    dicom_files = sorted([f for f in os.listdir(folder_us) if f.lower().endswith('.dcm')])
    numSlices = len(dicom_files)
    all_images = []
    num_images = 0
    structs = getStructures()
    sliceNum =0
    seedPos = getSeedPos(planPath)
    seedPositions = getSeedPos(planPath)

    '''over_images = overlay_dcm_dose_on_us(
        us_folder= folder_us,
        dose_file= dosePath,
       
        alpha = 0.35,
        colormap= 'jet',
        dose_threshold_percent = 5.0,
           # e.g. prescription dose 145 Gy      
    ) '''

    over_images = overlay_calc_dose_on_us(
        us_folder = folder_us,
                                          
        dose_file = dosePath, 
        plan_file = planPath,  
        alpha=0.3, 
        colormap='jet',
        isolines = True
        )
        
    '''  img_with_dose = overlay_dcm_dose_on_us(
        us_ds=ds,
        dose_ds=dose_ds,
        img_pil=img,
        slice =  sliceNum,
        alpha=0.3,           # semi-transparent
        colormap='jet'        # or 'hot', 'coolwarm', etc.
    )'''
        # overlay_calc_dose_on_us(us_ds, dose_file, plan_file, img_pil, slice, alpha=0.0, colormap='jet')
    '''img_with_dose = overlay_calc_dose_on_us(
        us_ds=ds,
        dose_file='DO001RG.dcm',
        plan_file = 'PL001RG.dcm',
        img_pil=img,
        slice =  sliceNum,
        alpha=0.3,           # semi-transparent
        colormap='jet'        # or 'hot', 'coolwarm', etc.
    )'''

    for image in over_images:
        
        drawSeeds(seedPos, image, sliceNum) 
        if sliceNum < len(structs):
            drawPoly(structs[sliceNum], image, color = 'red')
        buffer = BytesIO()
        image.save(buffer, format='PNG')
        #img.save(buffer, format='PNG')
        img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        all_images.append((img_base64, "one"))
        
        
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



def load_dicom_series(folder_path: str) -> List[sitk.Image]:
    """
    Load a folder of DICOM ultrasound slices as a list of 2D SimpleITK images.
    """
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(folder_path)
    reader.SetFileNames(dicom_names)
    reader.SetLoadPrivateTags(True)
    
    # Load as separate 2D images instead of forcing a 3D volume
    images_2d = []
    for fname in dicom_names:
        img = sitk.ReadImage(fname)
        # Ensure it's 2D
        if img.GetDimension() == 3 and img.GetSize()[2] == 1:
            img = img[:, :, 0]
        images_2d.append(img)
    
    return images_2d

#path = finders.find(os.path.join('files', 'SS001RG.dcm'))

def overlay_dcm_dose_on_us(
    us_folder: str,
    dose_file: str,
    slice_indices: Optional[List[int]] = None,
    alpha: float = 0.45,
    colormap: str = 'jet',
    dose_threshold_percent: float = 5.0,
    dose_norm_ref: Optional[float] = None,      # e.g. prescription dose 145 Gy
    output_folder: Optional[str] = None,
    isolines = True
) -> List[Image.Image]:
    """
    Overlay RTDOSE on a series of ultrasound DICOM images using SimpleITK.
    
    Parameters:
        us_folder:          folder containing US DICOM files
        dose_file:          path to RTDOSE DICOM file
        slice_indices:      optional - which dose slices to use (default: match order)
        alpha:              transparency of dose overlay
        colormap:           matplotlib colormap name
        dose_threshold_percent: hide dose below this % of max
        dose_norm_ref:      dose value to normalize to (e.g. prescription)
        output_folder:      if set, saves blended images as PNG
    
    Returns:
        List of PIL RGB images with dose overlay
    """
    # ── 1. Load ultrasound images ──────────────────────────────────────
    us_images = load_dicom_series(us_folder)
    print(f"Loaded {len(us_images)} ultrasound slices")

    # ── 2. Load RTDOSE ──────────────────────────────────────────────────
    dose_img = sitk.ReadImage(dose_file)
    dose_array = sitk.GetArrayFromImage(dose_img)  # shape: (nz, ny, nx)
    
    # Apply dose scaling
    ds = sitk.ReadImage(dose_file)  # pydicom not needed if metadata in SimpleITK
    dose_scaling = float(ds.GetMetaData("3004|000e")) if "3004|000e" in ds.GetMetaDataKeys() else 1.0
    dose_array = dose_array.astype(np.float32) * dose_scaling

    # ── 3. Prepare output list ──────────────────────────────────────────
    blended_images = []


    # If no slice indices provided, try to match number of US images
    if slice_indices is None:
        slice_indices = list(range(min(len(us_images), dose_array.shape[0])))

    # ── 4. Process each ultrasound slice ────────────────────────────────
    us_ds = pyd.dcmread(finders.find(os.path.join("Images","Richard_Gough", filename)))
    slice = 0
    for filename in os.listdir(us_folder):
        us_sitk = sitk.ReadImage(finders.find(os.path.join("Images","Richard_Gough", filename)))
        
        if slice >= dose_array.shape[0]:
            print(f"Warning: slice {slice} out of range for dose volume")
            continue

        if us_sitk.GetDimension() == 3 and us_sitk.GetSize()[2] == 1:
            us_sitk = us_sitk[:, :, 0]

        # Get dose slice as 2D SimpleITK image
        extractor = sitk.ExtractImageFilter()
        extractor.SetSize([dose_img.GetSize()[0], dose_img.GetSize()[1], 0])
        extractor.SetIndex([0, 0, slice])
        extractor.SetDirectionCollapseToStrategy(sitk.ExtractImageFilter.DIRECTIONCOLLAPSETOGUESS)

        dose_2d_sitk = extractor.Execute(dose_img)

        delta_x_tag = "0018|602c"   # Physical Delta X
        delta_y_tag = "0018|602e"   # Physical Delta Y

        delta_x_cm = None
        delta_y_cm = None

        # Check if the tags exist in metadata
        
        delta_x_cm = us_ds.PhysicalDeltaX
        delta_y_cm = us_ds.PhysicalDeltaY

        us_spacing_mm = [delta_x_cm * 10, delta_y_cm * 10]

        

        print("Dose2D dimension:", dose_2d_sitk.GetDimension())
        print("Dose2D direction length:", len(dose_2d_sitk.GetDirection()))
        print("Dose2D spacing:", dose_2d_sitk.GetSpacing())

        # ── 5. Resample dose to ultrasound geometry ─────────────────────
        resampler = sitk.ResampleImageFilter()
        resampler.SetOutputSpacing(us_spacing_mm)
        #resampler.SetReferenceImage(us_sitk)
        resampler.SetOutputOrigin(us_sitk.GetOrigin())          # match US position
        resampler.SetOutputDirection(us_sitk.GetDirection())
        resampler.SetSize(us_sitk.GetSize())
        resampler.SetInterpolator(sitk.sitkLinear)
        resampler.SetTransform(sitk.Transform(2, sitk.sitkIdentity))
        resampler.SetDefaultPixelValue(0.0)

        
        print("=== MUST SEE THESE VALUES ===")
        print("dose_2d_sitk.GetDimension()  :", dose_2d_sitk.GetDimension())     # MUST be 2
        print("dose_2d_sitk.GetSize()       :", dose_2d_sitk.GetSize())         # e.g. (120, 120)
        print("dose_2d_sitk.GetDirection() :", dose_2d_sitk.GetDirection())     # MUST have exactly 4 numbers
        print("us_sitk.GetDimension()      :", us_sitk.GetDimension())         # MUST be 2
        print("us_sitk.GetDirection()      :", us_sitk.GetDirection())

        dose_resampled_sitk = resampler.Execute(dose_2d_sitk)

        
        #dose_resampled_sitk = resampler.Execute(dose_2d_sitk)

        # Get numpy array
        dose_resampled_np = sitk.GetArrayFromImage(dose_resampled_sitk)

        # ── 6. Normalize dose for visualization ─────────────────────────
        if dose_norm_ref is None:
            dose_norm_ref = np.percentile(dose_resampled_np[dose_resampled_np > 0], 99.0)
        
        dose_norm = dose_resampled_np / dose_norm_ref
        dose_norm = np.clip(dose_norm, 0, 2.0)  # usually cap at 200%

        # ── 7. Create color overlay ─────────────────────────────────────
        cmap = cm.get_cmap(colormap)
        rgba = cmap(dose_norm)  # shape (h, w, 4)
        
        # Apply threshold and alpha
        mask = dose_norm >= (dose_threshold_percent / 100.0)
        rgba[..., 3] = np.where(mask, alpha, 0.0)

        # To uint8 RGBA
        overlay_rgba = (rgba * 255).astype(np.uint8)

        # ── 8. Blend with ultrasound ────────────────────────────────────
        us_np = sitk.GetArrayFromImage(us_sitk)
        # Handle grayscale → RGB
        if len(us_np.shape) == 2:
            us_rgb = np.stack([us_np] * 3, axis=-1)
        else:
            us_rgb = us_np

        us_pil = Image.fromarray(us_rgb.astype(np.uint8))
        overlay_pil = Image.fromarray(overlay_rgba, mode='RGBA')

        # Alpha composite
        blended = Image.alpha_composite(us_pil.convert('RGBA'), overlay_pil)
        blended_rgb = blended.convert('RGB')

        isoImage = draw_isodose_lines_single_image(us_np,
        dose_resampled_np,
        reference_dose=145.0,
        levels=[30, 50, 70, 90, 100, 120, 150]
        )
        if isolines:
            blended_images.append(isoImage)
        else:
            blended_images.append(blended_rgb)

        blended_images.append(blended_rgb)

        # Optional: save
        if output_folder:
            os.makedirs(output_folder, exist_ok=True)
            blended_rgb.save(os.path.join(output_folder, f"overlay_{i:03d}.png"))

       
        slice += 1

    return blended_images





def getKerma(plan_file):
    file = finders.find(os.path.join("files", plan_file))
    plan_ds = pyd.dcmread(file)
    for src in plan_ds.SourceSequence:
        kermaRate = src.ReferenceAirKermaRate
    return kermaRate

def getDoseGrid(dFile):
    dfile = finders.find(os.path.join("Files", dFile))
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
    return gx, gy, gz

def interp_gL(rs, gL_r, gL_val):
        # rs: (Nseeds, Npoints) -> (Nseeds, Npoints)
        out = cp.zeros_like(rs)
        for i in range(1, len(gL_r)):
            mask = (rs >= gL_r[i-1]) & (rs < gL_r[i])
            t = (rs - gL_r[i-1]) / (gL_r[i] - gL_r[i-1])
            out = cp.where(mask, gL_val[i-1] + t * (gL_val[i] - gL_val[i-1]), out)
        # Beyond 10 cm: 1/r² tail
        out = cp.where(rs >= 10.0, gL_val[-1], out)
        return out

def interp_aniso(rs, ani_r, ani_val):
        out = cp.zeros_like(rs)
        for i in range(1, len(ani_r)):
            mask = (rs >= ani_r[i-1]) & (rs < ani_r[i])
            t = (rs - ani_r[i-1]) / (ani_r[i] - ani_r[i-1])
            out = cp.where(mask, ani_val[i-1] + t * (ani_val[i] - ani_val[i-1]), out)
        out = cp.where(rs >= 5.0, ani_val[-1], out)
        out = cp.where(rs <= 0.5, ani_val[0], out)
        return out   

    
    # 1D anisotropy φ_an(r): simple fit, broadcasts

def calcDose(seed_pos, grid_x, grid_y, grid_z, Kerma):
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
    #print(f"points shape: {points.shape}")
    #print(f"X grid size:  {len(X)}")


    if isinstance(seed_pos, list):
        seeds = cp.asarray(np.array(seed_pos), dtype=cp.float32)
    else:
        seeds = cp.asarray(seed_pos, dtype=cp.float32)
    
    pos = seeds[:, 0:3]  # (Nseeds, 3)
    #dir_vec = seeds[:, 3:6]  # (Nseeds, 3)

    Nseeds, Npoints = pos.shape[0], points.shape[0]

    points_exp = points[None, :, :]  # (1, Npoints, 3)
    pos_exp = pos[:, None, :]        # (Nseeds, 1, 3)
    
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
    
    gL = interp_gL(r, gL_r, gL_val)
    aniso = interp_aniso(r, ani_r, ani_val)
    dose = cp.sum(Kerma * Lambda * 1/r**2 * gL * aniso * 2057, axis=0)
    dose = dose/100
    #print(f"dose : {dose}")
    #print(f"dose points : {dose.shape}")

    dose3d = dose.reshape((len(grid_x), len(grid_y), len(grid_z)))

    #print(f"dose3d : {dose3d}")
    dose3d = np.transpose(dose3d,(2,0,1))

    dose3d *= 1.0

    #print(f"dose3d processed: {dose3d}")

    '''print(f"max calc dose: {cp.max(dose3d):.2f} Gy")
    print(f"min calc dose: {cp.min(dose3d):.2f} Gy")
    print(f"mean calc dose: {cp.mean(dose3d):.2f} Gy")
    print(f"num points: {cp.shape(dose3d)[0]*cp.shape(dose3d)[1]*cp.shape(dose3d)[2]}")'''
    

    return dose3d.get()

def overlay_calc_dose_on_us(us_folder, 
    dose_file, 
    plan_file,  
    alpha=0.3, 
    colormap='jet',

    dose_threshold_percent: float = 5.0,
    dose_norm_ref: Optional[float] = None, 
    isolines = True
    ):
     
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
    dfile = finders.find(os.path.join("Files", dose_file))
    dose_ds = pyd.dcmread(dfile)
    dose_sitk = sitk
    dose_array = dose_ds.pixel_array.astype(float) * dose_ds.DoseGridScaling
    
    
    #dose_slice = np.rot90(dose_slice, k=1)  # often need rotation/flip to match orientation
    dose_offsets = np.array(dose_ds.GridFrameOffsetVector)

    # 2. Dose grid geometry
    dose_origin = np.array(dose_ds.ImagePositionPatient)
    dose_spacing = np.array(dose_ds.PixelSpacing)
    dose_orientation = np.array(dose_ds.ImageOrientationPatient)
    Kerma = getKerma(plan_file)
    seeds = getSeedPos(plan_file)
    gx, gy, gz = getDoseGrid(dose_file)
    calc_dose = calcDose(seeds, gx, gy, gz, Kerma)
    
    print(f"calc dose shape {calc_dose.shape}")
    print(f"dicom dose shape {dose_array.shape}")
    #print(dose)
    # 3. Ultrasound geometry (reference for overlay)
    #err_dose_slice = np.abs(calc_dose_slice - dose_slice)
    #err_dose_slice = calc_dose_slice - dose_slice
    
    # 4. Simple resampling: scale dose to match US size (approximate alignment)

    
    
    #print("dose shape")
    # print(dose_array.shape) 
    
    us_ds = pyd.dcmread(finders.find(os.path.join("Images","Richard_Gough", "US001.dcm")))
    X_delta = float(us_ds.PhysicalDeltaX)*10
    Y_delta = float(us_ds.PhysicalDeltaY)*10

    us_spacing_mm = [X_delta, Y_delta]
    
    print(f"delta X {X_delta}")
    #print(f"fisrt dose resampled shape {dose_resampled.shape}")

    # Clip/resample to match US shape (crop or pad if needed)
    dcm_dose_img = sitk.ReadImage(dose_file)

    blended_images = []
    slice = 0
    for filename in os.listdir(us_folder):
        us_sitk = sitk.ReadImage(finders.find(os.path.join("Images","Richard_Gough", filename)))
        dose_slice = dose_array[slice]
         
        dose_sitk = sitk.GetImageFromArray(dose_slice.astype(np.float32))
        if us_sitk.GetDimension() == 3 and us_sitk.GetSize()[2] == 1:
            us_sitk = us_sitk[:, :, 0]
        
        
        extractor = sitk.ExtractImageFilter()
        extractor.SetSize([dcm_dose_img.GetSize()[0], dcm_dose_img.GetSize()[1], 0])
        extractor.SetIndex([0, 0, slice])
        extractor.SetDirectionCollapseToStrategy(sitk.ExtractImageFilter.DIRECTIONCOLLAPSETOGUESS)

        dcm_dose_sitk = extractor.Execute(dcm_dose_img)
        dose_sitk.CopyInformation(dcm_dose_sitk)
        
        resampler = sitk.ResampleImageFilter()
        resampler.SetOutputSpacing(us_spacing_mm)
        #resampler.SetReferenceImage(us_sitk)
        resampler.SetOutputOrigin(us_sitk.GetOrigin())          # match US position
        resampler.SetOutputDirection(us_sitk.GetDirection())
        resampler.SetSize(us_sitk.GetSize())
        resampler.SetInterpolator(sitk.sitkLinear)
        resampler.SetTransform(sitk.Transform(2, sitk.sitkIdentity))
        resampler.SetDefaultPixelValue(0.0)

        dose_resampled = resampler.Execute(dose_sitk)
        
        
    
        #print(f"dose resampled values {dose_resampled}")
        dose_resampled_np = sitk.GetArrayFromImage(dose_resampled)

        # ── 6. Normalize dose for visualization ─────────────────────────
        if dose_norm_ref is None:
            dose_norm_ref = np.percentile(dose_resampled_np[dose_resampled_np > 0], 96.0)
        
        dose_norm = dose_resampled_np / dose_norm_ref
        dose_norm = np.clip(dose_norm, 0, 2.0)  # usually cap at 200%

        # ── 7. Create color overlay ─────────────────────────────────────
        cmap = cm.get_cmap(colormap)
        rgba = cmap(dose_norm)  # shape (h, w, 4)
        
        # Apply threshold and alpha
        mask = dose_norm >= (dose_threshold_percent / 100.0)
        rgba[..., 3] = np.where(mask, alpha, 0.0)

        # To uint8 RGBA
        overlay_rgba = (rgba * 255).astype(np.uint8)

        # ── 8. Blend with ultrasound ────────────────────────────────────
        us_np = sitk.GetArrayFromImage(us_sitk)
        # Handle grayscale → RGB
        if len(us_np.shape) == 2:
            us_rgb = np.stack([us_np] * 3, axis=-1)
        else:
            us_rgb = us_np

        us_pil = Image.fromarray(us_rgb.astype(np.uint8))
        overlay_pil = Image.fromarray(overlay_rgba, mode='RGBA')

        # Alpha composite
        blended = Image.alpha_composite(us_pil.convert('RGBA'), overlay_pil)
        blended_rgb = blended.convert('RGB')

        isoImage = draw_isodose_lines_single_image(us_np,
        dose_resampled_np,
        reference_dose=145.0,
        levels=[30, 50, 70, 90, 100, 120, 150]
        )
        if isolines:
            blended_images.append(isoImage)
        else:
            blended_images.append(blended_rgb)
            

        # Optional: save
        
       
        slice += 1

    return blended_images

def draw_isodose_lines_single_image(
    us_np,
    dose_resampled_np,
    reference_dose=145.0,
    levels=[30, 50, 70, 90, 100, 120, 150, 200],
    dpi = 100
):
    # Prepare data
    dose_percent = (dose_resampled_np / reference_dose) * 100.0
    
    if us_np.ndim == 3 and us_np.shape[0] == 1:
        us_np = us_np[0]

    # Create single figure + axes
    
    if us_np.ndim == 3 and us_np.shape[0] == 1:
            us_np = us_np[0]

    height, width = us_np.shape[:2] if us_np.ndim == 2 else us_np.shape[:2]

    fig = plt.figure(figsize=(width/dpi , height/dpi ))
    ax = fig.add_axes([0, 0, 1, 1])  # full image, no borders
    ax.imshow(
        us_np,
        cmap='gray',
        interpolation='none',
        origin='upper',                     # ← critical
        extent=[0, width, 0, height]        # ← defines coordinate range
    )
    # Draw multiple isodose lines
    cs = ax.contour(
        dose_percent,
        levels=levels,
        cmap='jet',
        linewidths=1.8,
        alpha=0.9,
        linestyles='solid',
        origin='upper',                     # ← must match imshow
        extent=[0, width, 0, height]        # ← must match imshow
    )

    # Add percentage labels
    ax.clabel(cs, inline=True, fontsize=9, fmt='%d%%')

    # Remove axes
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(False)

    # Convert plot to PIL image
    buf = BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)

    # Convert buffer to PIL Image
    annotated_pil = Image.open(buf)
    return annotated_pil


def getSeedPos(planFile):
    seedPos = []
    file = finders.find(os.path.join("files", planFile))
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
            bbox = [usx-7, usy-7, usx+7, usy+7]

            draw.ellipse(bbox, fill="green", outline=(0,0,0), width=1)
            
    return img

@require_POST
def handleSeed(request):
    USfile = finders.find(os.path.join("Images","Richard_Gough", 'US001.dcm'))
    planPath = finders.find(os.path.join('files', 'PL001RG.dcm'))
    us_ds = pyd.dcmread(USfile)
    plan_ds = pyd.dcmread( planPath)
    X_delta = float(us_ds.PhysicalDeltaX)
    Y_delta = float(us_ds.PhysicalDeltaY)

    data = json.loads(request.body)
    x = data['x'] * X_delta * 10
    y = data['y'] * Y_delta * 10
    slice = data['slice_index']*-5

    seeds = getSeedPos(planPath)

    for seed in seeds:
        if x > seed[0]-3 and  x < seed[0]+3 and y > seed[1]-3 and  y < seed[1]+3  and slice == seed[2]:
            print(seed)
   


    print(f"seed position x {x}")
    print(f"seed position y {y}")
    #for seed in seed_pos:
    #    if seed[2]
    return JsonResponse({
            'success': True,
            'message': f'Received click at pixel ({x:.1f}, {y:.1f})',
            # If you generate a new image:
            # 'new_overlay': 'data:image/png;base64,iVBORw0KGgo...' 
        })

def createSeedGrid(origin, num_slices):
    seedGrid = []
    print(f"num slices {num_slices}")
    for i in range(num_slices):
        for j in range(11):
            for k in range(13):
                seedGrid.append([origin[0]+ k*5, origin[1]+j*-5, i * -5])
    return seedGrid






seedGrid = []
@require_POST
def setOrigin(request):
   
    try:
        data = json.loads(request.body)
        slice_num = data.get('slice_num')
        origin_x = data.get('origin_x')
        origin_y = data.get('origin_y')
        origin = [origin_x, origin_y]
        image_width = data.get('image_width')
        image_height = data.get('image_height')
        seedGrid = createSeedGrid(origin, numSlices)
        print(f"seed grid {seedGrid}")

        return JsonResponse({
            'status': 'success',
            'slice': slice_num,
            'origin': [origin_x, origin_y]})

    except Exception as e:
        return JsonResponse({'status': 'error', 'message': str(e)}, status=400)

     


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




# Create your views here.

seedPos = []


def showPlan(request):
    seedPos = []
    file = finders.find(os.path.join("Files", planFile))
    plan = pyd.dcmread(file)
    print(f"Patient Name       : {plan.PatientName}")
    print(f"Patient ID         : {plan.PatientID}")
    print(f"Plan Name          : {plan.RTPlanLabel}")


    for src in plan.SourceSequence:
        #print(f"Source Type             : {src.SourceType}")                     # usually "POINT" or "LINE"
        
                      # e.g. I-125 OncoSeed, Pd-103, etc.
        print(f"Source Isotope          : {src.SourceIsotopeName}")
        kermaRate = src.ReferenceAirKermaRate

    
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
    dfile = finders.find(os.path.join("Files", doseFile))
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
    start_time = time.time()
    dose = calcDose(seedPos, gx, gy, gz, kermaRate)

    end_time = time.time()

    print(f"Calculation took: {end_time - start_time:.4f} seconds")

    #dfile = finders.find(os.path.join("Files/DO001.dcm"))
    plan_dose = getDose(dfile)
    getDoseErr(dose, plan_dose)

    #print(f"final calced dose: {dose}")
    #calcDoseCPU(file, dfile)
    return render(request, 'home.html', {})



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

class Plan:

    def __init__(self):
        self.planName
        self.SeedPositions
        
    
















    


