import re

from django.core import validators
from django.utils.deconstruct import deconstructible
from django.utils.translation import gettext_lazy as _


@deconstructible
class UnicodeContactNumberValidator(validators.RegexValidator):
    regex = r"^(\+91[\-\s]?)?[0]?(91)?[789]\d{9}$"
    message = _(
        "Enter a valid contact number!, "
        "Number length is 10 digits long."
    )
    flags = 0


@deconstructible
class UnicodeNameValidator(validators.RegexValidator):
    regex = r"^[A-Za-z]+(?: [A-Za-z]+)?$"
    message = _(
        "Enter a valid name!,"
        "It should not contain any special character."
    )
    flags = 0
