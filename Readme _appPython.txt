Advanced Data Analysis Web Application

An advanced web application for comprehensive data analysis, visualization, and machine learning.

Features:
- Data upload and preprocessing
- Advanced data cleaning and transformation
- Interactive data visualization
- Machine learning model training and evaluation
- Time series analysis
- Sentiment analysis
- Custom analysis requests
- PDF report generation
- Real-time chat interface for data insights

Prerequisites:
- Python 3.8+
- pip
- Virtual environment tool (e.g., venv)

Installation:
1. Clone the repository:
   git clone  https://github.com/Ki55n/Data-Fusion
   cd advanced-data-analysis-app

2. Create and activate a virtual environment:
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`

3. Install required packages:
   pip install -r requirements.txt

4. Download additional resources:
   python -m spacy download en_core_web_sm
   python -m nltk.downloader punkt

Configuration:
1. Create a .env file in the project root:
   GOOGLE_API_KEY=your_actual_api_key_here  (for personal reasons we cannot provide these keys)
   JWT_SECRET_KEY=your_secure_jwt_secret_key_here

2. Set up required directories:
   mkdir uploads processed static visualizations mlruns

Usage:
1. Start the application:
   python app.py

2. Open http://localhost:5000 in your web browser.

3. Upload your data file (CSV, Excel, or TXT).

4. Use the various analysis tools:
   - Data cleaning and transformation
   - Visualization generation
   - Machine learning model training
   - Advanced analytics
   - Custom analysis and report generation

5. Interact with the chat interface for insights and custom analyses.

6. Generate and download PDF reports.

Project Structure:
- app.py: Main application file
- static/: Static files (CSS, JavaScript)
- templates/: HTML templates
- uploads/: Uploaded data files
- processed/: Processed data files
- visualizations/: Generated visualization images
- mlruns/: MLflow experiment tracking

Troubleshooting:
- Update pip if you have installation issues:
  pip install --upgrade pip
- Ensure you have the correct TensorFlow version for your system.
- Verify Spacy and NLTK language models are installed.
- Check application logs for error messages.

Contributing:
1. Fork the repository
2. Create your feature branch (git checkout -b feature/AmazingFeature)
3. Commit your changes (git commit -m 'Add some AmazingFeature')
4. Push to the branch (git push origin feature/AmazingFeature)
5. Open a Pull Request

Please adhere to PEP 8 style guidelines and include appropriate tests.

License:
Distributed under the MIT License. See LICENSE file for more information.

Acknowledgments:
- Flask (https://flask.palletsprojects.com/)
- Scikit-learn (https://scikit-learn.org/)
- MLflow (https://mlflow.org/)
- Plotly (https://plotly.com/)
- TensorFlow (https://www.tensorflow.org/)

This application uses various Python libraries for data processing, analysis, and visualization. Key components include:

Data Processing:
- pandas: Data manipulation and analysis
- numpy: Numerical computing
- scikit-learn: Machine learning algorithms
- scipy: Scientific computing
- statsmodels: Statistical modeling

Visualization:
- matplotlib: 2D plotting
- seaborn: Statistical data visualization
- plotly: Interactive visualizations

Machine Learning:
- tensorflow: Deep learning
- keras: High-level neural networks API
- prophet: Time series forecasting

Natural Language Processing:
- nltk: Natural language toolkit
- spacy: Advanced NLP
- textblob: Simplified text processing

Web Framework:
- Flask: Web application framework
- Flask-SocketIO: Real-time communication
- Flask-JWT-Extended: JWT authentication

Reporting:
- reportlab: PDF generation

Other Utilities:
- joblib: Lightweight pipelining in Python
- chardet: Character encoding detection
- fuzzywuzzy: Fuzzy string matching
- pycountry: ISO country databases

The application integrates these libraries to provide a comprehensive suite of data analysis tools, from basic preprocessing to advanced machine learning models and interactive visualizations. It's designed to be extensible, allowing for easy addition of new features and analysis techniques.

For any questions, issues, or contributions, please open an issue or pull request on the GitHub repository.
