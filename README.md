# Bank Account Extractor

A smart financial analysis tool for extracting and analyzing bank statements from PDF files. Built with Flask, Python, and modern web technologies.

## 🚀 Features

### Core Functionality
- **PDF Bank Statement Extraction**: Automatically extract transaction data from various bank statement formats
- **Multi-Bank Support**: HDFC, Kotak, and generic bank statement parsing
- **Smart Analysis**: AI-powered transaction categorization and pattern recognition
- **Export Options**: Excel, XML, and JSON export formats
- **Date Range Filtering**: Extract transactions within specific date ranges

### Web Application
- **User Authentication**: Secure login with Google OAuth and traditional credentials
- **Responsive Design**: Mobile-first design with Bootstrap 5
- **Dark Mode**: Toggle between light and dark themes with persistence
- **Real-time Validation**: Form validation with instant feedback
- **Accessibility**: WCAG compliant with screen reader support

### Advanced Features
- **Recurring Transaction Detection**: Identify subscription payments and regular debits
- **Salary Credit Analysis**: Pattern recognition for salary deposits
- **High-Value Transaction Monitoring**: Flag transactions above configurable thresholds
- **Monthly Balance Tracking**: Historical balance analysis
- **Admin Dashboard**: User management and system monitoring

## 🛠️ Technology Stack

### Backend
- **Python 3.8+**: Core application logic
- **Flask**: Web framework
- **SQLAlchemy**: Database ORM
- **SQLite**: Database (configurable for production)
- **Google OAuth**: Authentication integration

### Frontend
- **Bootstrap 5**: Responsive UI framework
- **Bootstrap Icons**: Icon library
- **Vanilla JavaScript**: Modern ES6+ features
- **CSS3**: Advanced styling with animations

### PDF Processing
- **pdfplumber**: PDF text extraction
- **tabula-py**: Table extraction
- **Pandas**: Data manipulation and analysis

## 📁 Project Structure

```
Bank account extracter/
├── app.py                          # Main Flask application
├── bank_statement_extractor.py     # Core extraction logic
├── extractor/                      # Extraction modules
│   ├── __init__.py
│   ├── analysis.py                 # Transaction analysis
│   ├── exporters.py                # Export functionality
│   ├── extractor.py                # Main extractor class
│   ├── parsers/                    # Bank-specific parsers
│   │   ├── detect.py               # Bank detection
│   │   ├── generic.py              # Generic parser
│   │   ├── hdfc.py                 # HDFC Bank parser
│   │   └── kotak.py                # Kotak Bank parser
│   └── utils.py                    # Utility functions
├── models/                         # Database models
│   └── user.py                     # User model
├── templates/                      # HTML templates
│   ├── base.html                   # Base template with navigation
│   ├── home.html                   # Landing page
│   ├── index.html                  # Dashboard
│   ├── login.html                  # Login page
│   ├── admin_dashboard.html        # Admin interface
│   └── results.html                # Results display
├── static/                         # Static assets
│   ├── style.css                   # Main stylesheet
│   ├── js/                         # JavaScript files
│   │   └── utils.js                # Utility functions
│   └── images/                     # Images and icons
├── uploads/                        # File upload directory
├── outputs/                        # Generated output files
├── logs/                           # Application logs
└── requirements.txt                # Python dependencies
```

## 🚀 Getting Started

### Prerequisites
- Python 3.8 or higher
- pip package manager
- Git

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd "Bank account extracter"
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

5. **Initialize database**
   ```bash
   python -c "from app import init_db; init_db()"
   ```

6. **Run the application**
   ```bash
   python app.py
   ```

7. **Access the application**
   Open your browser and navigate to `http://localhost:5000`

## 🔧 Configuration

### Environment Variables
Create a `.env` file in the root directory:

```env
FLASK_SECRET_KEY=your-secret-key-here
GOOGLE_CLIENT_ID=your-google-oauth-client-id
GOOGLE_CLIENT_SECRET=your-google-oauth-client-secret
FLASK_ENV=development
```

### Google OAuth Setup
1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project or select existing one
3. Enable Google+ API
4. Create OAuth 2.0 credentials
5. Add authorized redirect URIs:
   - `http://localhost:5000/login/google/authorized` (development)
   - `https://yourdomain.com/login/google/authorized` (production)

## 📖 Usage

### Basic Usage
1. **Upload PDF**: Select a bank statement PDF file
2. **Set Parameters**: Choose date range and optional password
3. **Extract Data**: Click extract to process the statement
4. **View Results**: Review extracted transactions and analysis
5. **Export**: Download results in Excel, XML, or JSON format

### Advanced Features
- **Transaction Categorization**: Automatic categorization of transactions
- **Pattern Recognition**: Identify recurring payments and salary credits
- **Balance Verification**: Cross-check running balances
- **Custom Analysis**: Configure thresholds and analysis parameters

## 🔒 Security Features

- **Password Protection**: Secure PDF password handling
- **User Authentication**: Multi-factor authentication support
- **Session Management**: Secure session handling
- **File Validation**: Strict file type and size restrictions
- **Input Sanitization**: Protection against injection attacks

## ♿ Accessibility Features

- **Screen Reader Support**: ARIA labels and semantic HTML
- **Keyboard Navigation**: Full keyboard accessibility
- **High Contrast Mode**: Support for high contrast displays
- **Reduced Motion**: Respects user motion preferences
- **Focus Indicators**: Clear focus states for all interactive elements

## 🎨 UI/UX Features

- **Responsive Design**: Mobile-first approach
- **Dark Mode**: Toggle between light and dark themes
- **Smooth Animations**: CSS transitions and micro-interactions
- **Loading States**: Visual feedback during operations
- **Toast Notifications**: Non-intrusive user feedback

## 🧪 Testing

### Run Tests
```bash
python -m pytest tests/
```

### Test Coverage
```bash
python -m pytest --cov=app tests/
```

## 📊 Performance

- **File Processing**: Optimized PDF parsing algorithms
- **Database Queries**: Efficient SQL queries with proper indexing
- **Caching**: Smart caching for frequently accessed data
- **Async Processing**: Background processing for large files

## 🚀 Deployment

### Production Setup
1. **Use Production WSGI Server**
   ```bash
   pip install gunicorn
   gunicorn -w 4 -b 0.0.0.0:8000 app:app
   ```

2. **Configure Reverse Proxy** (Nginx example)
   ```nginx
   server {
       listen 80;
       server_name yourdomain.com;
       
       location / {
           proxy_pass http://127.0.0.1:8000;
           proxy_set_header Host $host;
           proxy_set_header X-Real-IP $remote_addr;
       }
   }
   ```

3. **Set Production Environment**
   ```bash
   export FLASK_ENV=production
   export FLASK_SECRET_KEY=your-production-secret
   ```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Bootstrap Team**: For the excellent UI framework
- **Flask Community**: For the robust web framework
- **Python Community**: For the amazing ecosystem
- **Open Source Contributors**: For various libraries and tools

## 📞 Support

- **Documentation**: [Wiki](link-to-wiki)
- **Issues**: [GitHub Issues](link-to-issues)
- **Discussions**: [GitHub Discussions](link-to-discussions)
- **Email**: support@bankextractor.com

## 🔄 Changelog

### Version 2.0.0 (Current)
- ✨ Enhanced dark mode with persistence
- ♿ Improved accessibility features
- 📱 Better mobile responsiveness
- 🔒 Enhanced security features
- 🎨 Modern UI improvements
- 🚀 Performance optimizations

### Version 1.0.0
- 🎉 Initial release
- 📄 Basic PDF extraction
- 🔐 User authentication
- 📊 Transaction analysis
- 📤 Export functionality

---

**Made with ❤️ by the Bank Account Extractor Team**
