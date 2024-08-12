from datetime import datetime


def gst_certificate_path(instance, filename):
    return f'thumb/categories/{datetime.now().year}/{datetime.now().month}/{datetime.now().day}/{instance}-{filename}'


def business_profile_pic_path(instance, filename):
    return f'business/pic/{datetime.now().year}/{datetime.now().month}/{datetime.now().day}/{instance}-{filename}'


def product_images_path(instance, filename):
    return f'product/images/{datetime.now().year}/{datetime.now().month}/{datetime.now().day}/{instance}-{filename}'

def product_videos_path(instance, filename):
    return f'product/videos/{datetime.now().year}/{datetime.now().month}/{datetime.now().day}/{instance}-{filename}'


def catalogue_images_path(instance, filename):
    return f'catalogue/images/{datetime.now().year}/{datetime.now().month}/{datetime.now().day}/{instance}-{filename}'
