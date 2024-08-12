from . import models
from rest_framework import serializers
from .mailservice import SendMail

mail = SendMail()


class UserRegisterSerializer(serializers.ModelSerializer):
    """user register serializer"""
    password = serializers.CharField(write_only=True)

    def validate(self, attrs):
        """validate attributes is valid or not"""
        print('attrs: ', attrs)
        return attrs

    def create(self, validated_data):
        user = models.User.objects.create_user(
            name=validated_data['name'],
            email=validated_data['email'].lower(),
            contact_number=0,
            password=validated_data['password'],
        )
        models.UserDetails.objects.create(user_id=user.id)
        models.UserExtra.objects.create(user_id=user.id)
        # otp = mail.generate_otp()
        # models.OTPVerification.objects.create(user_id=user.id, email_id=user.email, phone_number=user.contact_number,
        #                                       otp=otp)

        # mail.send_otp(user, otp, "email")
        return user

    class Meta:
        model = models.User
        fields = ["id", "name", "email", "contact_number", "password"]

class UserLoginSerializer(serializers.ModelSerializer):
    """user login serializer"""
    email = serializers.EmailField(max_length=255)

    class Meta:
        model = models.User
        fields = ["email", "password"]
        

class CategorySerializer(serializers.ModelSerializer):
    """Category Serializer"""
    
    class Meta:
        model = models.Category
        fields = '__all__'
        
class ProductSerializer(serializers.ModelSerializer):
    """product serializer"""
    
    class Meta:
        model = models.Product
        fields = ["id", "title", "price", 'category', "extra_data"]
