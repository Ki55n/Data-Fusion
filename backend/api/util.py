import random
import string
import re
from django.core.exceptions import ValidationError


def validate_name(name):
    pattern = r"^[A-Za-z]+(?: [A-Za-z]+)?$"
    if re.match(pattern, name) is not None:
        return name
    else:
        raise ValidationError("Enter a valid name.")


def validate_contact_number(number):
    pattern = r"^(?:\+91|0)?[6-9]\d{9}$"
    if re.match(pattern, number) is not None:
        return number
    else:
        raise ValidationError("Enter a valid contact number.")


def generate_unique_id(length):
    characters = string.ascii_uppercase + string.digits
    unique_id = ''.join(random.choice(characters) for _ in range(length))
    return unique_id
