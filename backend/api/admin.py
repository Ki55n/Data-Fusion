from django.contrib import admin
from . import models


# Register your models here.
@admin.register(models.User, models.UserExtra, models.Category, models.Product)
class APIModelsAdmin(admin.ModelAdmin):
    pass