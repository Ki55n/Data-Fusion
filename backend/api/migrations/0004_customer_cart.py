# Generated by Django 4.2 on 2024-05-15 09:20

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('api', '0003_product_quantity_category_alter_product_category'),
    ]

    operations = [
        migrations.CreateModel(
            name='Customer',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('name', models.CharField(max_length=250)),
                ('extra_data', models.JSONField(default=dict)),
                ('created_on', models.DateTimeField(auto_now_add=True)),
                ('updated_on', models.DateTimeField(auto_now=True)),
                ('orders', models.ManyToManyField(related_name='customer_orders', to='api.order')),
            ],
            options={
                'verbose_name': 'Customer',
            },
        ),
        migrations.CreateModel(
            name='Cart',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('quantity', models.IntegerField(default=1)),
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('updated_at', models.DateTimeField(auto_now=True)),
                ('customer', models.OneToOneField(on_delete=django.db.models.deletion.CASCADE, related_name='customer_cart', to='api.customer')),
                ('product', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='cart_products', to='api.product')),
            ],
            options={
                'verbose_name': 'Cart',
            },
        ),
    ]