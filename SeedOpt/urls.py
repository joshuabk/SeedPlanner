
from django.urls import path
from . import views

app_name = 'SeedOpt'

urlpatterns = [
    path('', views.showPlan, name='showPlan')

]