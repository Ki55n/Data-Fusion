import json
import os

from django.contrib.auth import authenticate
from django.shortcuts import get_object_or_404
from rest_framework import generics, permissions
from rest_framework import status
from rest_framework.response import Response
from rest_framework_simplejwt.tokens import RefreshToken
from django.core.exceptions import ObjectDoesNotExist

from . import models
from . import serializers
from .mailservice import SendMail
from .pagination import DatamindPagination
from .util import generate_unique_id

mail = SendMail()


# Create your views here.
def get_tokens_for_user(user):
    refresh = RefreshToken.for_user(user)
    return {
        'refresh': str(refresh),
        'access': str(refresh.access_token),
    }


class UserRegisterView(generics.CreateAPIView):
    serializer_class = serializers.UserRegisterSerializer
    permission_classes = [permissions.AllowAny]

    def post(self, request, *args, **kwargs):
        serializer = serializers.UserRegisterSerializer(data=request.data)
        if not serializer.is_valid():
            messages: dict = {}
            for key, value in dict(serializer.errors).items():
                messages[key] = value[0]
            return Response(data={'messages': messages, 'status': {'msg': 'failed', 'code': 220}})
        user = serializer.save()
        token = get_tokens_for_user(user)
        return Response({'token': token, 'message': 'Registration Successful.',
                         'status': {'code': 200, 'msg': 'success'}}, status=status.HTTP_200_OK)


class UserLoginView(generics.CreateAPIView):
    serializer_class = serializers.UserLoginSerializer
    permission_classes = [permissions.AllowAny]

    def post(self, request, *args, **kwargs):
        serializer = serializers.UserLoginSerializer(data=request.data)
        if not serializer.is_valid():
            messages: dict = {}
            for key, value in dict(serializer.errors).items():
                messages[key] = value[0]
            return Response(data={'messages': messages, 'status': {'msg': 'failed', 'code': 220}})
        email = serializer.data.get('email')
        password = serializer.data.get('password')
        user = authenticate(email=email, password=password)
        if user is not None:
            token = get_tokens_for_user(user)
            return Response(
                {'token': token, 'message': 'Login Successful.', 'status': {'msg': 'success', 'code': 200}},
                status=status.HTTP_200_OK)
        else:
            return Response({'message': 'Email or Password is not Valid',
                             'status': {'msg': 'success', 'code': 230}}, status=status.HTTP_404_NOT_FOUND)


class CreateCategoryView(generics.CreateAPIView):
    permission_classes = [permissions.IsAuthenticated]

    def post(self, request, *args, **kwargs):
        name = request.data["name"]
        if name == "":
            return Response(data={
                'message': 'Please provide category name.',
                'status': {
                    'code': 230,
                    'msg': 'failed'
                }
            })
        category = models.Category.objects.create(user_id=request.user.id, name=name)
        serializer = serializers.CategorySerializer(category)
        return Response(data={
            'message': 'Successfully category created.',
            'status': {
                'code': 200,
                'msg': 'success'
            },
            'results': serializer.data
        })


class CategoryListView(generics.ListAPIView):
    permission_classes = [permissions.IsAuthenticated]

    def get(self, request, *args, **kwargs):
        queryset = models.Category.objects.all().order_by('id').reverse()
        serializer = serializers.CategorySerializer(queryset, many=True)
        return Response(data={'status': {'code': 200, 'msg': 'success'}, 'message': 'Successfully retrieved',
                              'results': serializer.data})


class ProductCreateView(generics.CreateAPIView):
    permission_classes = [permissions.IsAuthenticated]

    def post(self, request, *args, **kwargs):
        products = request.data["products"]
        if products == "" or products == []:
            return Response(data={'status': {'code': 220, 'msg': 'failed'},
                                  'message': 'Please send product details to create products.'})

        user_extra = models.UserExtra.objects.get(user_id=request.user.id)

        try:
            for product in products:
                if product['category_id'] == "":
                    return Response(
                        data={'status': {'code': 230, 'msg': 'failed'}, 'message': 'Category is not available.'})
                if not models.Category.objects.filter(id=product['category_id']).exists():
                    return Response(
                        data={'status': {'code': 230, 'msg': 'failed'}, 'message': 'Category is not available.'})
                p = models.Product.objects.create(
                    title=product['title'],
                    description=product['description'],
                    price=product['price'],
                    quantity=product['quantity'],
                    category_id=product['category_id'],
                    extra_data=product)

                user_extra.products.add(p)
                user_extra.save()
            return Response(data={'status': {'code': 200, 'msg': 'success'}, 'message': 'Successfully inserted.'})
        except Exception as e:
            return Response(data={'status': {'code': 230, 'msg': 'failed'}, 'message': e})


class ProductUpdateView(generics.UpdateAPIView):
    permission_classes = [permissions.IsAuthenticated]

    def patch(self, request, *args, **kwargs):
        product_id = request.data["product_id"]
        if product_id == "":
            return Response(data={
                'status': {
                    'code': 230,
                    'msg': 'failed'
                },
                'message': 'Please provide product id.',
            })
        product = models.Product.objects.get(pk=product_id)
        product.title = request.data["extra_data"]["title"]
        product.price = request.data["extra_data"]["price"]
        product.description = request.data["extra_data"]["description"]
        product.quantity = request.data["extra_data"]["quantity"]
        product.category_id = request.data["extra_data"]["category_id"]
        product.extra_data = request.data["extra_data"]
        product.save()
        serializer = serializers.ProductSerializer(product)
        return Response(data={
            'status': {
                'code': 200,
                'msg': 'success'
            },
            'message': 'Successfully updated.',
            'results': serializer.data
        })


class ProductListView(generics.ListAPIView):
    permission_classes = [permissions.IsAuthenticated]

    def get(self, request, *args, **kwargs):
        paginator = DatamindPagination()
        page_size = request.GET.get('page-size')
        if page_size is not None and page_size != "":
            paginator.page_size = int(page_size)

        queryset = models.UserExtra.objects.get(user_id=request.user.id).products.all().order_by('id').reverse()
        result_queryset = paginator.paginate_queryset(queryset, request)
        serializer = serializers.ProductSerializer(result_queryset, many=True)
        s = {'code': 200, 'msg': 'success'}
        return paginator.get_paginated_response(data=serializer.data, msg="Successfully retrieved.", status=s)


class CreateCustomer(generics.CreateAPIView):
    permission_classes = [permissions.IsAuthenticated]

    def post(self, request, *args, **kwargs):
        name = request.data["name"]
        if name == "":
            return Response(data={
                'message': 'Please provide customer name.',
                'status': {'code': 230, 'msg': 'failed'}
            })
        customer = models.Customer.objects.create(name=name, extra_data=request.data)
        serializer = serializers.CustomerSerializer(customer)
        return Response(data={
            'message': 'Successfully created customer.',
            'status': {'code': 200, 'msg': 'success'},
            'results': serializer.data
        })


class CustomersListView(generics.ListAPIView):
    permission_classes = [permissions.IsAuthenticated]

    def get(self, request, *args, **kwargs):
        paginator = DatamindPagination()
        page_size = request.GET.get('page-size')
        if page_size is not None and page_size != "":
            paginator.page_size = int(page_size)

        queryset = models.Customer.objects.all().order_by('id').reverse()
        result_queryset = paginator.paginate_queryset(queryset, request)
        serializer = serializers.CustomerSerializer(result_queryset, many=True)
        s = {'code': 200, 'msg': 'success'}
        return paginator.get_paginated_response(data=serializer.data, msg="Successfully retrieved.", status=s)


class CustomerCartView(generics.RetrieveAPIView):
    permission_classes = [permissions.IsAuthenticated]

    def get(self, request, *args, **kwargs):
        customer_id = request.data["customer_id"]

        if customer_id == "":
            return Response(data={
                'message': 'Please provide customer id.',
                'status': {'code': 230, 'msg': 'failed'}
            })

        customer = models.Customer.objects.get(pk=customer_id)
        cart_products = models.Cart.objects.filter(customer_id=customer_id)

        serializer = serializers.CartSerializer(cart_products, many=True)
        return Response(data={
            'status': {
                'code': 200,
                'msg': 'success',
            },
            'message': 'Successfully retrieved.',
            'results': serializer.data
        })


class CustomerDetailView(generics.RetrieveAPIView):
    permission_classes = [permissions.IsAuthenticated]

    def get(self, request, *args, **kwargs):
        customer_id = request.data["customer_id"]
        queryset = models.Customer.objects.get(id=customer_id)
        serializer = serializers.CustomerSerializer(queryset)
        return Response(data={
            'status': {
                'code': 200,
                'msg': 'success'
            },
            'message': 'Successfully retrieved',
            'results': serializer.data
        })


class OrdersListView(generics.ListAPIView):
    permission_classes = [permissions.IsAuthenticated]

    def get(self, request, *args, **kwargs):
        paginator = DatamindPagination()
        customer_id = request.GET.get("customer_id")
        page_size = request.GET.get('page-size')
        if page_size is not None and page_size != "":
            paginator.page_size = int(page_size)

        queryset = models.Customer.objects.get(id=customer_id).orders.all().order_by('id').reverse()
        result_queryset = paginator.paginate_queryset(queryset, request)
        serializer = serializers.OrderSerializer(result_queryset, many=True)
        s = {'code': 200, 'msg': 'success'}
        return paginator.get_paginated_response(data=serializer.data, msg="Successfully retrieved.", status=s)
