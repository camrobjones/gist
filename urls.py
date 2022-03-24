"""Gist URL configurations"""

from django.urls import path
from gist import views


app_name = 'gist'
urlpatterns = [
    path('', views.home),
    path('analyse', views.analyse),
    path('wordnet', views.wordnet_home),
    path('synset_data', views.synset_data),
    path('search_lemma', views.search_lemma)
]
