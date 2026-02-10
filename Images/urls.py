from django.urls import path
from . import views


app_name = 'Images'

urlpatterns = [
    path('images/', views.showUS, name='showUS'),
    path('images/handleSeed/', views.handleSeed, name='handleSeed'),
    path('images/setOrigin/', views.setOrigin, name='setOrigin')

]