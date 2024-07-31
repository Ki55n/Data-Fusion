from django.conf import settings
from rest_framework import pagination
from rest_framework.response import Response


class DatamindPagination(pagination.PageNumberPagination):

    def get_paginated_response(self, data, msg=None, status=None):
        if self.page.has_next():
            next_page = self.page.next_page_number()
        else:
            next_page = None

        if self.page.has_previous():
            previous_page = self.page.previous_page_number()
        else:
            previous_page = None
        return Response({
            'nex_page': next_page,
            'pre_page': previous_page,
            'total_pages': self.page.paginator.num_pages,
            'count': self.page.paginator.count,
            'status': status,
            'message': msg,
            'data': data
        })
