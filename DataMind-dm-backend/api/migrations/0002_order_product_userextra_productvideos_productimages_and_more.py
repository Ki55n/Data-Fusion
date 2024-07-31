# Generated by Django 4.2 on 2024-05-11 19:58

import api.filepath
from django.conf import settings
from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('api', '0001_initial'),
    ]

    operations = [
        migrations.CreateModel(
            name='Order',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('Order id', models.CharField(max_length=200)),
                ('is_paid', models.BooleanField(default=True)),
                ('extra_data', models.JSONField(null=True)),
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('updated_at', models.DateTimeField(auto_now=True)),
            ],
            options={
                'verbose_name': 'Orders',
            },
        ),
        migrations.CreateModel(
            name='Product',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('title', models.CharField(max_length=1000)),
                ('description', models.TextField()),
                ('price', models.CharField(max_length=100)),
                ('category', models.CharField(max_length=100)),
                ('is_active', models.BooleanField(default=False)),
                ('message', models.CharField(blank=True, help_text='If it is not active (Optional)', max_length=300, null=True)),
                ('extra_data', models.JSONField(null=True)),
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('updated_at', models.DateTimeField(auto_now=True)),
            ],
            options={
                'verbose_name': 'Products',
            },
        ),
        migrations.CreateModel(
            name='UserExtra',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('updated_at', models.DateTimeField(auto_now=True)),
                ('orders', models.ManyToManyField(related_name='user_orders', to='api.order')),
                ('products', models.ManyToManyField(related_name='user_products', to='api.product')),
                ('user', models.OneToOneField(on_delete=django.db.models.deletion.CASCADE, related_name='user_extra', to=settings.AUTH_USER_MODEL)),
            ],
            options={
                'verbose_name': 'User Extras (Mapped Data)',
            },
        ),
        migrations.CreateModel(
            name='ProductVideos',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('videos', models.FileField(upload_to=api.filepath.product_images_path)),
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('updated_at', models.DateTimeField(auto_now=True)),
                ('product', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='product_videos', to='api.product')),
            ],
            options={
                'verbose_name': 'Product Videos',
            },
        ),
        migrations.CreateModel(
            name='ProductImages',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('images', models.ImageField(upload_to=api.filepath.product_images_path)),
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('updated_at', models.DateTimeField(auto_now=True)),
                ('product', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='product_images', to='api.product')),
            ],
            options={
                'verbose_name': 'Product Images',
            },
        ),
        migrations.CreateModel(
            name='OrderProducts',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('quantity', models.IntegerField(default=1)),
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('updated_at', models.DateTimeField(auto_now=True)),
                ('order', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='order_products', to='api.order')),
                ('products', models.ForeignKey(on_delete=django.db.models.deletion.DO_NOTHING, related_name='products_order', to='api.product')),
            ],
            options={
                'verbose_name': 'Order Products',
            },
        ),
    ]
