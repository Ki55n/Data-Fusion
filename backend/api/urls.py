from rest_framework_simplejwt import views as jwt_views
from django.urls import path
from . import views

urlpatterns = [
    # """ Authentication """,
    path('register', views.UserRegisterView.as_view(), name="register"),
    path('login', views.UserLoginView.as_view(), name="login"),
    path('token/refresh', jwt_views.TokenRefreshView.as_view(), name="token-refresh"),
    path('logout', jwt_views.TokenBlacklistView.as_view(), name="logout"),

    # """ Products """
    path('product/create', views.ProductCreateView.as_view(), name="create-product"),
    path('product/update', views.ProductUpdateView.as_view(), name="update-product"),
    path('product/get', views.ProductListView.as_view(), name="get-product"),

    # """ Category """
    path('category/create', views.CreateCategoryView.as_view(), name="create_category"),
    path('category/get', views.CategoryListView.as_view(), name="get_categories"),

    # """ Customer """
    path('customer/create', views.CreateCustomer.as_view(), name="create-customer"),
    path('customer/detait', views.CustomerDetailView.as_view(), name="detail-customer"),
    path('customer/get', views.CustomersListView.as_view(), name="get_customers"),
    path('customer/cart', views.CustomerCartView.as_view(), name="customer-cart"),

    # """ Orders """
    path('order/get', views.OrdersListView.as_view(), name="get_orders"),
]
