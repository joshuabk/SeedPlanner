from django.shortcuts import render

import pydicom as pyd
from pydicom.dataset import Dataset
from pprint import pprint
import numpy as np
from django.contrib.staticfiles import finders
import os

# Create your views here.


def showPlan(request):
    file = finders.find(os.path.join("PL001.dcm"))
    plan = pyd.dcmread(file)
    print(f"Patient Name       : {plan.PatientName}")
    print(f"Patient ID         : {plan.PatientID}")
    print(f"Plan Name          : {plan.RTPlanLabel}")


    for src in plan.SourceSequence:
        print(f"Source Type             : {src.SourceType}")                     # usually "POINT" or "LINE"
        
                      # e.g. I-125 OncoSeed, Pd-103, etc.
        print(f"Source Isotope          : {src.SourceIsotopeName}")

    for app in enumerate(plan.ApplicationSetupSequence):
        #print(f"Air Kerma   : {app.ApplicationSetupType}")
        for channel in app[1].ChannelSequence:
            print(channel)
            print("channel printed")
            print(f"Number of Control Points   : {channel.NumberOfControlPoints}")
            for cp in enumerate(channel.BrachyControlPointSequence):  # every other is duplicate
                print(cp[1])
                print("print cp")
                pos = cp[1].ControlPointIndex
                
                print(f"control point          :{pos}")

    return render(request, 'home.html', {})

