# DataFusion - Backend

This project is a backend system for managing products, categories, catalogues, and user authentication. It is built with Django and provides APIs for the following functionalities:

Product Management: Create, update, and delete products.
Category Management: Manage product categories.
Catalogue Management: Create and manage catalogues.
User Authentication: Handle user registration, login, and authentication.
Seller Management: Manage seller details.
Features
Django Framework: Leverages the power of Django for rapid development and scalability.
RESTful API: Provides a clean and straightforward API for frontend integration.
User Authentication: Secure user authentication with Django's built-in authentication system.
Product Management: CRUD operations for products, categories, and catalogues.
Seller Management: Manage seller details, including profile information and product listings.
Prerequisites
Python 3.x
Django 4.2 or higher
Django Rest Framework (DRF)
MySQL (or any other preferred database)

#Setup Django Project
Requirements: Python: >=3.8
Step 1
- Create Virtual environment :  'python -m venv backend-env'
  
- Activate virtual environment:  'source backend-env/bin/activate'

Step 2 : Install Django and Setup Project
- 'pip install django==4.2 (Install django)'
- 'pip install -m requirements.txt'
- 'python manage.py makemigrations' (create table (generate objects of DB))
- 'python manage.py migrate' (migrate database)
- 'python managte.py runserver' (Run project)
