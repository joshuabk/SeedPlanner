
from django.urls import path
from . import views
urlpatterns = [
    path('', views.showPlan, name='showPlan')

]