import math
import random

# from django.conf import settings
from django.core.mail import send_mail
from dmproj import settings


class SendMail:

    @staticmethod
    def generate_otp():
        num_str = '0123456789'
        OTP = ""
        length = len(num_str)
        for i in range(5):
            OTP += num_str[math.floor(random.random() * length)]
        return OTP

    @staticmethod
    def send_otp(user, otp, of):
        if of == "gst":
            subject = 'GST Verification'
            message = (f'Hi {user.name}, here is your otp is {otp} for GST verification.'
                       f'Don\'t share this otp with anyone.')
        else:
            subject = 'Email Verification'
            message = (f'Hi {user.name}, thank you for registering on example.com\n'
                       f'To verify email enter otp is {otp}.')

        print('email: ', user.email)

        email_from = settings.EMAIL_HOST_USER
        recipient_list = [user.email, ]
        send_mail(subject, message, email_from, recipient_list, fail_silently=False)
