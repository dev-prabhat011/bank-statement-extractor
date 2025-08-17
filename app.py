# app.py (Updated with BuildError fix, Login Redirect fix, Pricing Route) 

from flask import (Flask, render_template, request, send_from_directory,
                   redirect, url_for, flash, session, abort, jsonify, send_file) # Removed Blueprint as it wasn't used
import os
import time
import uuid
from werkzeug.utils import secure_filename
import traceback
import json
import base64
import io
import zipfile
from extractor.exporters import export_excel, export_xml
from dotenv import load_dotenv
from flask_login import LoginManager, UserMixin, login_user, logout_user, current_user, login_required
import logging # Ensure logging is imported
from werkzeug.security import generate_password_hash, check_password_hash
import sqlite3
from datetime import timedelta, datetime
from flask_sqlalchemy import SQLAlchemy

# --- Google OAuth Libraries ---
try:
    from google_auth_oauthlib.flow import Flow
    import google.auth.transport.requests
    from google.oauth2 import id_token
    import requests as req # Use alias
except ImportError:
    print("ERROR: Missing Google OAuth libraries. Run: pip install Flask-Login google-auth google-auth-oauthlib requests python-dotenv")
    import sys
    sys.exit(1)

# --- Load your StatementExtractor ---
try:
    # Using the new modular StatementExtractor wrapper
    from extractor.extractor import StatementExtractor
except ImportError as imp_err:
    print(f"ERROR: Cannot import StatementExtractor: {imp_err}")
    print("Ensure extractor/extractor.py is in the correct path and saved with that name.")
    import sys
    sys.exit(1)

# --- Load environment variables ---
load_dotenv() # Loads variables from .env file in the root directory

# --- Flask App Setup ---
app = Flask(__name__)
app.debug = True  # Enable debug mode
# IMPORTANT: Use a strong, random secret key from environment variable or config
app.secret_key = os.environ.get("FLASK_SECRET_KEY", os.urandom(24))
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['OUTPUT_FOLDER'] = 'outputs'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024 # 16MB
app.config['ALLOWED_EXTENSIONS'] = {'pdf'}

# --- Health Check Endpoint ---
@app.route('/health')
def health_check():
    """Health check endpoint for Docker and load balancers."""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.utcnow().isoformat(),
        'service': 'Bank Statement Extractor',
        'version': '1.0.0'
    }), 200

# --- Logging Setup ---
# Ensure logs directory exists
logs_dir = os.path.join(app.root_path, 'logs')
os.makedirs(logs_dir, exist_ok=True)

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # Log to console
        logging.FileHandler(os.path.join(logs_dir, 'flask_app.log'))  # Log to file
    ]
)
logger = logging.getLogger(__name__)

# --- SQLAlchemy Setup ---
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///app.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SQLALCHEMY_ECHO'] = True  # Log all SQL queries
db = SQLAlchemy(app)

# --- Flask-Login User Class ---
class User(UserMixin, db.Model):
    __tablename__ = 'user'
    id = db.Column(db.String(36), primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=True)
    profile_pic = db.Column(db.String(500), nullable=True)
    is_admin = db.Column(db.Boolean, default=False)
    role = db.Column(db.String(20), default='user')
    active = db.Column(db.Boolean, default=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    last_login = db.Column(db.DateTime, nullable=True)
    banks = db.relationship('Bank', backref='user', lazy=True)
    subscriptions = db.relationship('Subscription', backref='user', lazy=True)
    extractions = db.relationship('Extraction', backref='user', lazy=True)
    activities = db.relationship('Activity', backref='user', lazy=True)

    def __init__(self, id_, name, email, profile_pic=None, password=None, is_admin=False, role='user', active=True):
        self.id = str(id_)  # Ensure ID is string
        self.name = str(name)  # Ensure name is string
        self.email = str(email)  # Ensure email is string
        self.profile_pic = str(profile_pic) if profile_pic else "https://via.placeholder.com/30/007bff/ffffff?text=U"
        self.password = str(password) if password else None
        self.is_admin = is_admin
        self.role = role
        self.active = active
        self.created_at = datetime.utcnow()

    def get_id(self):
        return str(self.id)

    @staticmethod
    def get(user_id):
        return User.query.get(str(user_id))

    @staticmethod
    def get_by_email(email):
        return User.query.filter_by(email=str(email)).first()
        
    def to_dict(self):
        return {
            'id': self.id,
            'name': self.name,
            'email': self.email,
            'role': self.role,
            'active': self.active,
            'last_login': self.last_login.isoformat() if self.last_login else None,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }

# --- Other Models ---
class Bank(db.Model):
    __tablename__ = 'bank'
    id = db.Column(db.String(36), primary_key=True)
    user_id = db.Column(db.String(36), db.ForeignKey('user.id'), nullable=False)
    bank_name = db.Column(db.String(100), nullable=False)
    account_number = db.Column(db.String(50), nullable=True)
    account_type = db.Column(db.String(50), nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    last_updated = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    extractions = db.relationship('Extraction', backref='bank', lazy=True)

    def __init__(self, id_, user_id, bank_name, account_number=None, account_type=None):
        self.id = str(id_)
        self.user_id = str(user_id)
        self.bank_name = str(bank_name)
        self.account_number = str(account_number) if account_number else None
        self.account_type = str(account_type) if account_type else None

    def to_dict(self):
        return {
            'id': self.id,
            'user_id': self.user_id,
            'bank_name': self.bank_name,
            'account_number': self.account_number,
            'account_type': self.account_type,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'last_updated': self.last_updated.isoformat() if self.last_updated else None
        }

class Subscription(db.Model):
    __tablename__ = 'subscription'
    id = db.Column(db.String(36), primary_key=True)
    user_id = db.Column(db.String(36), db.ForeignKey('user.id'), nullable=False)
    plan = db.Column(db.String(50), nullable=False)
    status = db.Column(db.String(20), default='active')
    start_date = db.Column(db.DateTime, default=datetime.utcnow)
    end_date = db.Column(db.DateTime, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    last_updated = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def __init__(self, id_, user_id, plan, status='active', end_date=None):
        self.id = str(id_)
        self.user_id = str(user_id)
        self.plan = str(plan)
        self.status = str(status)
        self.end_date = end_date

    def to_dict(self):
        return {
            'id': self.id,
            'user_id': self.user_id,
            'plan': self.plan,
            'status': self.status,
            'start_date': self.start_date.isoformat() if self.start_date else None,
            'end_date': self.end_date.isoformat() if self.end_date else None,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'last_updated': self.last_updated.isoformat() if self.last_updated else None
        }

class Extraction(db.Model):
    __tablename__ = 'extraction'
    id = db.Column(db.String(36), primary_key=True)
    user_id = db.Column(db.String(36), db.ForeignKey('user.id'), nullable=False)
    bank_id = db.Column(db.String(36), db.ForeignKey('bank.id'), nullable=False)
    file_name = db.Column(db.String(255), nullable=False)
    status = db.Column(db.String(20), nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    
    def __init__(self, id_, user_id, bank_id, file_name, status):
        self.id = str(id_)
        self.user_id = str(user_id)
        self.bank_id = str(bank_id)
        self.file_name = str(file_name)
        self.status = str(status)
    
    def to_dict(self):
        return {
            'id': self.id,
            'user_id': self.user_id,
            'bank_id': self.bank_id,
            'file_name': self.file_name,
            'status': self.status,
            'timestamp': self.timestamp.isoformat()
        }

class Activity(db.Model):
    __tablename__ = 'activity'
    id = db.Column(db.String(36), primary_key=True)
    user_id = db.Column(db.String(36), db.ForeignKey('user.id'), nullable=False)
    action = db.Column(db.String(50), nullable=False)
    details = db.Column(db.Text)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    
    def __init__(self, id_, user_id, action, details=None):
        self.id = str(id_)
        self.user_id = str(user_id)
        self.action = str(action)
        self.details = str(details) if details else None
    
    def to_dict(self):
        return {
            'id': self.id,
            'user_id': self.user_id,
            'action': self.action,
            'details': self.details,
            'timestamp': self.timestamp.isoformat()
        }

# --- Flask-Login Setup ---
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = "login"  # Redirect to login page if @login_required fails
login_manager.login_message = "Please log in to access this page."
login_manager.login_message_category = "info"
login_manager.session_protection = "strong"  # Enable session protection

# Configure session settings
app.config['SESSION_TYPE'] = 'filesystem'
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(days=7)  # Set session lifetime to 7 days
app.config['REMEMBER_COOKIE_DURATION'] = timedelta(days=7)  # Set remember cookie duration
app.config['REMEMBER_COOKIE_SECURE'] = False  # Allow HTTP for development
app.config['REMEMBER_COOKIE_HTTPONLY'] = True  # Prevent JavaScript access to cookie
app.config['SESSION_COOKIE_SECURE'] = False  # Allow HTTP for development
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'

@login_manager.user_loader
def load_user(user_id):
    """Loads user object for Flask-Login session management."""
    try:
        return User.get(user_id)
    except Exception as e:
        print(f"⚠️ Error loading user {user_id}: {str(e)}")
        return None

# --- Database Initialization ---
def init_db():
    """Initialize the database with required tables and admin user."""
    try:
        # Create database directory if it doesn't exist
        db_uri = app.config['SQLALCHEMY_DATABASE_URI']
        if db_uri.startswith('sqlite:///'):
            db_path = db_uri.replace('sqlite:///', '')
            db_dir = os.path.dirname(db_path)
            if db_dir and not os.path.exists(db_dir):
                os.makedirs(db_dir, exist_ok=True)
                print(f"Created database directory: {db_dir}")
        else:
            print(f"Database URI: {db_uri}")

        # Create all tables
        with app.app_context():
            db.create_all()
            print("✅ Created all database tables")

            # Check if admin user exists
            admin = User.query.filter_by(email='admin@example.com').first()
            if not admin:
                # Create admin user
                admin = User(
                    id_=str(uuid.uuid4()),
                    name='Admin User',
                    email='admin@example.com',
                    password=generate_password_hash('admin123'),
                    is_admin=True,
                    role='admin',
                    active=True
                )
                db.session.add(admin)
                print("✅ Created admin user")

                # Create sample bank for admin
                bank = Bank(
                    id_=str(uuid.uuid4()),
                    user_id=admin.id,
                    bank_name='Sample Bank',
                    account_number='1234567890',
                    account_type='Savings'
                )
                db.session.add(bank)
                print("✅ Created sample bank for admin")

                # Create subscription for admin
                subscription = Subscription(
                    id_=str(uuid.uuid4()),
                    user_id=admin.id,
                    plan='premium',
                    status='active',
                    end_date=datetime.utcnow() + timedelta(days=30)
                )
                db.session.add(subscription)
                print("✅ Created subscription for admin")

                db.session.commit()
                print("✅ Committed all changes to database")
            else:
                print("ℹ️ Admin user already exists")

    except Exception as e:
        print(f"❌ Error initializing database: {str(e)}")
        raise

# Initialize database on startup
try:
    with app.app_context():
        init_db()
        print("✅ Database initialized successfully")
except Exception as e:
    print(f"⚠️ Warning: Database initialization failed: {e}")
    print("App will continue but some features may not work")

# Define absolute paths for folders relative to the app's root path
try:
    upload_folder_path = os.path.join(app.root_path, app.config['UPLOAD_FOLDER'])
    output_folder_path = os.path.join(app.root_path, app.config['OUTPUT_FOLDER'])
    os.makedirs(upload_folder_path, exist_ok=True)
    os.makedirs(output_folder_path, exist_ok=True)
    print(f"✅ Created directories: {upload_folder_path}, {output_folder_path}")
except Exception as e:
    print(f"⚠️ Warning: Failed to create directories: {e}")
    # Use fallback paths
    upload_folder_path = 'uploads'
    output_folder_path = 'outputs'

# --- Google OAuth Configuration ---
GOOGLE_CLIENT_ID = os.environ.get("GOOGLE_CLIENT_ID", None)
GOOGLE_CLIENT_SECRET = os.environ.get("GOOGLE_CLIENT_SECRET", None)
# Ensure this matches Google Console EXACTLY
GOOGLE_REDIRECT_URI = "http://127.0.0.1:5000/login/google/authorized" # Adjust port if needed

client_secrets = None
if GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET:
    client_secrets = {
        "web": {
            "client_id": GOOGLE_CLIENT_ID,
            "client_secret": GOOGLE_CLIENT_SECRET,
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
            "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
        }
    }
    print("✅ Google OAuth configured")
else:
    print("⚠️ Google OAuth Client ID or Secret not configured. Google Login disabled.")

SCOPES = ['https://www.googleapis.com/auth/userinfo.profile',
          'https://www.googleapis.com/auth/userinfo.email',
          'openid']
# Allow http for local development ONLY - REMOVE FOR PRODUCTION HTTPS
os.environ["OAUTHLIB_INSECURE_TRANSPORT"] = "1"

# --- Helper Functions ---
def allowed_file(filename):
    """Check if the file extension is allowed."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# --- Routes ---
@app.route('/favicon.ico')
def favicon():
    """Serve favicon to prevent 404 errors."""
    return '', 204  # No content response

@app.route('/')
def home_page():
    """Renders the home landing page."""
    logger.debug("Rendering home landing page")
    return render_template('home.html')

@app.route('/test')
def test_page():
    """Test route to verify template inheritance."""
    return render_template('test.html')

@app.route('/simple_test')
def simple_test_page():
    """Simple test route to verify template processing."""
    return render_template('simple_test.html', test_var='Hello from Flask!')

@app.route('/home_new')
def home_new_page():
    """Test route for the new home template."""
    return render_template('home_new.html')



@app.route('/index')
@login_required
def index():
    """Renders the main extractor upload form page."""
    logger.debug(f"Rendering extractor page for user: {current_user.id if current_user.is_authenticated else 'Anonymous'}")
    # Clear only file-related session data if needed
    session.pop('last_upload_path', None)
    session.pop('start_date', None)
    session.pop('end_date', None)
    session.pop('output_formats', None)
    return render_template('index.html') # Renders index.html template

# --- Google Login Routes ---

@app.route("/login/google")
def google_login():
    """Initiates the Google OAuth flow"""
    try:
        if not client_secrets:
            logger.error("Google OAuth not configured")
            flash("Google login is not configured", "error")
            return redirect(url_for("home_page"))

        # Create flow instance to manage the OAuth 2.0 Authorization Grant Flow steps.
        flow = Flow.from_client_config(
            client_config=client_secrets,
            scopes=SCOPES,
            redirect_uri=GOOGLE_REDIRECT_URI
        )

        # Generate URL for request to Google's OAuth 2.0 server.
        authorization_url, state = flow.authorization_url(
            access_type='offline',  # Enable refresh tokens
            include_granted_scopes='true',
            prompt='consent'  # Force consent screen to ensure refresh token
        )

        # Store the state so the callback can verify the auth server response.
        session["oauth_state"] = state
        logger.info(f"Starting Google OAuth flow for state: {state}")

        return redirect(authorization_url)

    except Exception as e:
        logger.error(f"Error initiating Google login: {str(e)}")
        flash("An error occurred during login", "error")
        return redirect(url_for("home_page"))

@app.route("/login/google/authorized")
def google_callback():
    """Handle Google OAuth callback"""
    try:
        if not client_secrets:
            logger.error("Google OAuth not configured")
            flash("Google login is not configured", "error")
            return redirect(url_for("home_page"))

        # Create flow instance to manage the OAuth 2.0 Authorization Grant Flow steps
        flow = Flow.from_client_config(
            client_config=client_secrets,
            scopes=SCOPES,
            redirect_uri=GOOGLE_REDIRECT_URI
        )

        # Get authorization code from Google
        code = request.args.get('code')
        if not code:
            logger.error("No authorization code received from Google")
            flash('Failed to authenticate with Google', 'error')
            return redirect(url_for('login'))

        # Exchange authorization code for credentials
        flow.fetch_token(code=code)
        credentials = flow.credentials

        # Get user info from Google
        session = flow.authorized_session()
        userinfo_response = session.get('https://www.googleapis.com/oauth2/v3/userinfo').json()
        
        if not userinfo_response or not userinfo_response.get('email'):
            logger.error("Failed to get user info from Google")
            flash('Failed to get user information from Google', 'error')
            return redirect(url_for('login'))

        # Extract user information
        email = userinfo_response['email']
        name = userinfo_response.get('name', email.split('@')[0])
        picture = userinfo_response.get('picture')

        try:
            # Check if user exists
            user = User.query.filter_by(email=email).first()
            if not user:
                # Create new user
                user = User(
                    id_=str(uuid.uuid4()),
                    name=name,
                    email=email,
                    password=generate_password_hash(str(uuid.uuid4())),  # Random password for OAuth users
                    profile_pic=picture,
                    is_admin=False,
                    role="user",
                    active=True
                )
                db.session.add(user)
                db.session.commit()
                logger.info(f"Created new user from Google OAuth: {email}")

            # Update user's last login
            user.last_login = datetime.utcnow()
            db.session.commit()

            # Log in the user
            login_user(user)
            logger.info(f"User logged in via Google OAuth: {email}")

            # Redirect to index (main dashboard) for regular users, admin_dashboard for admins
            if user.is_admin:
                return redirect(url_for('admin_dashboard'))
            return redirect(url_for('index'))

        except Exception as e:
            logger.error(f"Database error during Google OAuth: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            db.session.rollback()
            flash('An error occurred while processing your login. Please try again.', 'error')
            return redirect(url_for('login'))

    except Exception as e:
        logger.error(f"Error in Google OAuth callback: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        flash('An error occurred during Google authentication. Please try again.', 'error')
        return redirect(url_for('login'))

@app.route("/logout")
@login_required
def logout():
    """Logs the user out."""
    user_email = session.get('user_info', {}).get('email', 'Unknown User')
    logout_user()
    session.clear()
    flash('Successfully logged out.', 'info')
    logger.info(f"User {user_email} logged out.")
    return redirect(url_for('home_page'))

# --- End Google Login Routes ---

@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        if current_user.is_admin:
            return redirect(url_for('admin_dashboard'))
        return redirect(url_for('analyzer'))

    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        
        user = User.get_by_email(email)
        
        if user and check_password_hash(user.password, password):
            login_user(user, remember=True)
            session.permanent = True
            session['user_info'] = {
                'name': user.name,
                'email': user.email,
                'picture': user.profile_pic
            }
            flash('Successfully logged in!', 'success')
            
            # Check if user is admin and redirect accordingly
            if user.is_admin:
                return redirect(url_for('admin_dashboard'))
            else:
                next_page = request.args.get('next')
                return redirect(next_page if next_page else url_for('index'))
        else:
            flash('Invalid email or password', 'error')
            logger.warning(f"Failed login attempt for email: {email}")
    
    return render_template('login.html')

@app.route('/register', methods=['POST'])
def register():
    name = request.form.get('name')
    email = request.form.get('email')
    password = request.form.get('password')
    confirm_password = request.form.get('confirm_password')
    
    if password != confirm_password:
        flash('Passwords do not match', 'error')
        return redirect(url_for('home_page'))
    
    if User.get_by_email(email):
        flash('Email already exists', 'error')
        return redirect(url_for('home_page'))
    
    user_id = str(uuid.uuid4())
    hashed_password = generate_password_hash(password)
    
    try:
        user = User(
            id_=user_id,
            name=name,
            email=email,
            password=hashed_password
        )
        db.session.add(user)
        db.session.commit()
        flash('Registration successful! Please log in.', 'success')
    except Exception as e:
        db.session.rollback()
        logger.error(f"Database error during registration: {e}")
        flash('Registration failed. Please try again.', 'error')
    
    return redirect(url_for('login'))

# --- Pricing Page Route ---
@app.route('/pricing')
def pricing_page():
    """Renders the pricing page."""
    logger.debug("Rendering pricing page")
    # Assuming you have 'templates/pricing.html'
    return render_template('pricing.html')
# --- End Pricing Page Route ---


# --- Existing Upload/Password/Download Routes ---

@app.route('/upload', methods=['POST'])
@login_required # <<< Keep protected
def upload_file():
    """Handles file upload, processing, and rendering results or errors."""
    if 'file' not in request.files:
        flash('No file part')
        return redirect(url_for('index')) # Redirect back to extractor page

    file = request.files['file']
    if file.filename == '':
        flash('No selected file')
        return redirect(url_for('index')) # Redirect back to extractor page

    if file and allowed_file(file.filename):
        timestamp = int(time.time())
        filename = f"{timestamp}_{secure_filename(file.filename)}"
        file_path = os.path.join(upload_folder_path, filename)

        password = request.form.get('password', '') or None
        start_date = request.form.get('start_date')
        end_date = request.form.get('end_date')
        bank_name = request.form.get('bank_name', '') or None
        other_bank_name = request.form.get('other_bank_name', '') or None
        output_excel = 'excel' in request.form
        output_xml = 'xml' in request.form

        # Handle "Other" bank selection
        if bank_name == 'OTHER' and other_bank_name:
            bank_name = other_bank_name.upper()
            # Check if this bank already exists in the database
            existing_bank = Bank.query.filter_by(bank_name=bank_name).first()
            if not existing_bank:
                # Add the new bank to the database
                new_bank = Bank(
                    id=str(uuid.uuid4()),
                    user_id=current_user.id,
                    bank_name=bank_name
                )
                db.session.add(new_bank)
                db.session.commit()
                logger.info(f"Added new bank to database: {bank_name}")
                
                # Log activity
                activity = Activity(
                    id=str(uuid.uuid4()),
                    user_id=current_user.id,
                    action='add_bank',
                    details=f'Added new bank: {bank_name}'
                )
                db.session.add(activity)
                db.session.commit()

        if not start_date or not end_date:
            flash('Start Date and End Date are required.')
            return redirect(url_for('analyzer')) # Redirect back to extractor page

        output_formats = []
        if output_excel: output_formats.append('excel')
        if output_xml: output_formats.append('xml')
        if not output_formats: output_formats = ['excel', 'xml']

        try:
            logger.info(f"Saving uploaded file to: {file_path}")
            file.save(file_path)
            logger.info("File saved successfully.")

            extractor = StatementExtractor(
                file_path,
                start_date_str=start_date,
                end_date_str=end_date,
                password=password,
                debug=app.debug,
                bank_name=bank_name
            )

            logger.info("Calling extractor.extract_all()")
            # Ensure extractor returns 3 values and they are unpacked correctly
            transactions, account_info, analysis_results = extractor.extract_all()
            logger.info(f"Extractor finished. Transactions found: {len(transactions)}")


            base_name = os.path.splitext(filename)[0]
            result_files = []

            # Export logic
            if 'excel' in output_formats:
                if transactions: # Only export if transactions exist
                    excel_fname = f"{base_name}.xlsx"
                    excel_path = os.path.join(output_folder_path, excel_fname)
                    try:
                        export_excel(extractor, excel_path)
                        result_files.append(('Excel', excel_fname))
                    except Exception as export_err:
                         logger.error(f"Failed to export Excel: {export_err}", exc_info=True)
                         flash(f"Processing successful, but failed to create Excel file: {export_err}", 'warning')
                else:
                     logger.warning("Skipping Excel export: No transactions found.")

            if 'xml' in output_formats:
                 if transactions: # Only export if transactions exist
                    xml_fname = f"{base_name}.xml"
                    xml_path = os.path.join(output_folder_path, xml_fname)
                    try:
                        export_xml(extractor, xml_path)
                        result_files.append(('XML', xml_fname))
                    except Exception as export_err:
                        logger.error(f"Failed to export XML: {export_err}", exc_info=True)
                        flash(f"Processing successful, but failed to create XML file: {export_err}", 'warning')
                 else:
                      logger.warning("Skipping XML export: No transactions found.")

            # Log before render
            logger.info(f"Rendering results page. Transactions length: {len(transactions)}")

            return render_template('results.html',
                                   transactions=transactions,
                                   account_info=account_info,
                                   analysis_results=analysis_results, # Pass analysis results if needed
                                   result_files=result_files)

        except ValueError as ve:
             if os.path.exists(file_path):
                 try: os.remove(file_path)
                 except OSError as e: logger.error(f"Error removing file {file_path} after ValueError: {e}")
             logger.warning(f"Processing ValueError: {ve}")
             flash(f"Processing Error: {ve}")
             if "password" in str(ve).lower():
                  session['last_upload_path'] = file_path
                  session['start_date'] = start_date
                  session['end_date'] = end_date
                  session['output_formats'] = output_formats
                  return redirect(url_for('password_page'))
             else:
                  # Redirect to extractor page on other value errors
                  return redirect(url_for('analyzer')) # <<< FIXED: Redirect to extractor page

        except Exception as e:
             logger.error(f"Caught unexpected exception during upload processing for file {filename}", exc_info=True)
             if 'file_path' in locals() and os.path.exists(file_path):
                 try: os.remove(file_path)
                 except OSError as e_os: logger.error(f"Error removing file {file_path} after Exception: {e_os}")
             flash(f'An unexpected error occurred during processing. Please check logs or try again.')
             # Redirect to extractor page on unexpected errors
             return redirect(url_for('analyzer')) # <<< FIXED: Redirect to extractor page
    else:
        flash('Invalid file type. Only PDF files are allowed.')
        return redirect(url_for('analyzer')) # Redirect back to extractor page


@app.route('/password', methods=['GET', 'POST'])
@login_required # <<< Keep protected
def password_page():
    """Handles the password entry page and reprocessing with password."""
    file_path = session.get('last_upload_path')
    start_date = session.get('start_date')
    end_date = session.get('end_date')
    output_formats = session.get('output_formats', ['excel', 'xml'])

    if not file_path or not start_date or not end_date:
        flash('Session expired or invalid state. Please re-upload the file.')
        return redirect(url_for('analyzer')) # Go back to extractor page

    if request.method == 'POST':
        password = request.form.get('password', '')
        if not password:
            flash('Password cannot be empty.')
            return render_template('password.html')

        try:
            extractor = StatementExtractor(
                file_path,
                start_date_str=start_date,
                end_date_str=end_date,
                password=password,
                debug=app.debug
            )

            logger.info("Calling extractor.extract_all() from password route")
            # Ensure unpacking 3 values
            transactions, account_info, analysis_results = extractor.extract_all()
            logger.info(f"Reprocessing successful. Transactions length: {len(transactions)}")


            filename = os.path.basename(file_path)
            base_name = os.path.splitext(filename)[0]
            result_files = []

            # Export logic
            if 'excel' in output_formats:
                 if transactions:
                    excel_fname = f"{base_name}.xlsx"
                    excel_path = os.path.join(output_folder_path, excel_fname)
                    try:
                        extractor.export_to_excel(excel_path)
                        result_files.append(('Excel', excel_fname))
                    except Exception as export_err:
                        logger.error(f"Excel export failed: {export_err}", exc_info=True)
                        flash(f"Excel export failed, but other formats may be available: {export_err}", 'warning')
                 else:
                      logger.warning("Skipping Excel export: No transactions found.")

            if 'xml' in output_formats:
                 if transactions:
                    xml_fname = f"{base_name}.xml"
                    xml_path = os.path.join(output_folder_path, xml_fname)
                    try:
                        extractor.to_xml(xml_path)
                        result_files.append(('XML', xml_fname))
                    except Exception as export_err:
                        logger.error(f"XML export failed: {export_err}", exc_info=True)
                        flash(f"XML export failed, but other formats may be available: {export_err}", 'warning')
                 else:
                      logger.warning("Skipping XML export: No transactions found.")


            # Clear session data on success
            session.pop('last_upload_path', None)
            session.pop('start_date', None)
            session.pop('end_date', None)
            session.pop('output_formats', None)

            logger.info(f"Rendering results page from password route. Transactions length: {len(transactions)}")

            return render_template('results.html',
                                   transactions=transactions,
                                   account_info=account_info,
                                   analysis_results=analysis_results,
                                   result_files=result_files)

        except ValueError as ve:
            logger.warning(f"Processing ValueError in password route: {ve}")
            if "password" in str(ve).lower():
                flash('Incorrect password. Please try again.')
                return render_template('password.html') # Stay on password page
            else:
                flash(f"Processing Error: {ve}")
                 # Redirect to extractor page on other value errors
                return redirect(url_for('analyzer')) # <<< FIXED: Redirect to extractor page

        except Exception as e:
            logger.error(f"Caught unexpected exception during password processing for file {file_path}", exc_info=True)
            flash(f'An unexpected error occurred during processing. Please check logs or try again.')
            session.pop('last_upload_path', None)
            session.pop('start_date', None)
            session.pop('end_date', None)
            session.pop('output_formats', None)
            # Redirect to extractor page on unexpected errors
            return redirect(url_for('analyzer')) # <<< FIXED: Redirect to extractor page

    # GET request
    return render_template('password.html')


@app.route('/download/<filename>')
@login_required # <<< Keep protected
def download_file(filename):
    """Provides generated files for download."""
    try:
        safe_filename = secure_filename(filename)
        logger.info(f"Attempting to send file: {safe_filename} from {output_folder_path}")
        return send_from_directory(output_folder_path, # Use absolute path
                                 safe_filename,
                                 as_attachment=True)
    except FileNotFoundError:
        logger.error(f"Download requested for non-existent file: {filename}")
        abort(404, description="File not found")
    except Exception as e:
         logger.error(f"Error during file download for {filename}", exc_info=True)
         abort(500, description="Could not download file")


# --- Public API Endpoint for Salesforce Integration ---
@app.route('/api/extract', methods=['POST'])
def api_extract():
    """Public API to process a bank statement PDF and return Excel/XML as base64 in JSON.
    Accepts:
      - multipart/form-data with 'file' (PDF), and optional fields: password, start_date, end_date, bank_name
      - application/json with fields: file_base64 (PDF as base64), filename (optional), password, start_date, end_date, bank_name, output (list)
    Returns JSON with base64-encoded Excel and XML files and basic metadata.
    """
    try:
        # Helper to get param from either form or json
        def get_param(key, default=None):
            if request.is_json:
                data = request.get_json(silent=True) or {}
                return data.get(key, default)
            return request.form.get(key, default)

        # Parse input file
        temp_filename = None
        file_path = None
        if 'file' in request.files:
            uploaded = request.files['file']
            if uploaded.filename == '':
                return jsonify({'error': 'No file provided'}), 400
            if not allowed_file(uploaded.filename):
                return jsonify({'error': 'Only PDF files are allowed'}), 400
            timestamp = int(time.time())
            temp_filename = f"api_{timestamp}_{secure_filename(uploaded.filename)}"
            file_path = os.path.join(upload_folder_path, temp_filename)
            uploaded.save(file_path)
        elif request.is_json and (request.json or {}):
            data = request.get_json(silent=True) or {}
            file_b64 = data.get('file_base64')
            filename = secure_filename(data.get('filename') or 'statement.pdf')
            if not file_b64:
                return jsonify({'error': 'file_base64 is required in JSON body'}), 400
            try:
                pdf_bytes = base64.b64decode(file_b64)
            except Exception:
                return jsonify({'error': 'file_base64 is not valid base64'}), 400
            if not filename.lower().endswith('.pdf'):
                filename += '.pdf'
            timestamp = int(time.time())
            temp_filename = f"api_{timestamp}_{filename}"
            file_path = os.path.join(upload_folder_path, temp_filename)
            with open(file_path, 'wb') as f:
                f.write(pdf_bytes)
        else:
            return jsonify({'error': 'No file provided. Send multipart/form-data with file or JSON with file_base64'}), 400

        # Params
        password = get_param('password') or None
        start_date = get_param('start_date')
        end_date = get_param('end_date')
        bank_name = get_param('bank_name') or None
        output = get_param('output')  # e.g., ['excel','xml']
        if isinstance(output, str):
            try:
                output = json.loads(output)
            except Exception:
                output = [o.strip() for o in output.split(',') if o.strip()]
        if not output:
            output = ['excel', 'xml']

        # Validate required dates if your extractor requires them
        if not start_date or not end_date:
            # Optional: allow missing dates; adjust if extractor requires
            logger.info("API: start_date or end_date not provided; proceeding without strict validation.")

        # Run extractor
        extractor = StatementExtractor(
            file_path,
            start_date_str=start_date,
            end_date_str=end_date,
            password=password,
            debug=app.debug,
            bank_name=bank_name
        )
        transactions, account_info, analysis_results = extractor.extract_all()

        if not transactions:
            # Clean up temp file
            try:
                if file_path and os.path.exists(file_path):
                    os.remove(file_path)
            except Exception:
                pass
            return jsonify({'error': 'No transactions found in the statement'}), 422

        # Prepare outputs in temp files
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        resp = {
            'status': 'success',
            'meta': {
                'extraction_id': getattr(extractor, 'unique_id', str(uuid.uuid4())),
                'bank_name': account_info.get('bank_name') or bank_name,
                'account_number': account_info.get('account_number'),
                'transactions_count': len(transactions)
            },
            'files': {}
        }

        if 'excel' in output:
            excel_fname = f"{base_name}.xlsx"
            excel_path = os.path.join(output_folder_path, excel_fname)
            extractor.export_to_excel(excel_path)
            with open(excel_path, 'rb') as f:
                excel_b64 = base64.b64encode(f.read()).decode('utf-8')
            resp['files']['excel'] = {
                'filename': excel_fname,
                'content_base64': excel_b64,
                'content_type': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
            }
            # Optional cleanup of excel file can be added here

        if 'xml' in output:
            xml_fname = f"{base_name}.xml"
            xml_path = os.path.join(output_folder_path, xml_fname)
            extractor.to_xml(xml_path)
            with open(xml_path, 'rb') as f:
                xml_b64 = base64.b64encode(f.read()).decode('utf-8')
            resp['files']['xml'] = {
                'filename': xml_fname,
                'content_base64': xml_b64,
                'content_type': 'application/xml'
            }

        # Cleanup uploaded temp file (keep outputs as optional)
        try:
            if file_path and os.path.exists(file_path):
                os.remove(file_path)
        except Exception:
            pass

        return jsonify(resp), 200

    except Exception as e:
        logger.error(f"API extract error: {e}", exc_info=True)
        return jsonify({'error': 'Internal server error'}), 500


@app.errorhandler(413)
def request_entity_too_large(error):
    """Handles file uploads exceeding the size limit."""
    flash('File too large! Maximum file size is 16MB.')
     # Redirect to extractor page if file is too large for upload attempt
    return redirect(url_for('analyzer')), 413 # <<< FIXED: Redirect to extractor page

@app.route('/clear_session')
def clear_session():
    """Clears the user session."""
    session.clear()
    flash("Session cleared.")
    return redirect(url_for('home_page')) # Redirect to home page

# Admin routes
@app.route('/admin')
@login_required
def admin_dashboard():
    if not current_user.is_admin:
        flash('Access denied. Admin privileges required.', 'danger')
        return redirect(url_for('home_page'))
    
    # Get statistics
    total_users = User.query.count()
    total_subscriptions = Subscription.query.filter_by(status='active').count()
    total_extractions = Extraction.query.count()
    total_banks = Bank.query.count()
    
    # Get recent activities
    recent_activities = Activity.query.order_by(Activity.timestamp.desc()).limit(10).all()
    
    # Get data for each tab
    banks = Bank.query.all()
    users = User.query.all()
    subscriptions = Subscription.query.all()
    extractions = Extraction.query.order_by(Extraction.timestamp.desc()).limit(50).all()
    
    return render_template('admin_dashboard.html',
                         total_users=total_users,
                         total_subscriptions=total_subscriptions,
                         total_extractions=total_extractions,
                         total_banks=total_banks,
                         recent_activities=recent_activities,
                         banks=banks,
                         users=users,
                         subscriptions=subscriptions,
                         extractions=extractions)

# Admin routes for adding new items
@app.route('/admin/user/add', methods=['POST'])
@login_required
def add_user():
    if not current_user.is_admin:
        return jsonify({'error': 'Access denied'}), 403
    
    data = request.json
    name = data.get('name')
    email = data.get('email')
    password = data.get('password')
    role = data.get('role', 'user')
    
    if not all([name, email, password]):
        return jsonify({'error': 'Name, email, and password are required'}), 400
    
    # Check if user already exists
    if User.query.filter_by(email=email).first():
        return jsonify({'error': 'User with this email already exists'}), 400
    
    # Hash password
    hashed_password = generate_password_hash(password)
    
    # Create new user
    new_user = User(
        id=str(uuid.uuid4()),
        name=name,
        email=email,
        password=hashed_password,
        role=role,
        is_admin=(role == 'admin'),
        active=True
    )
    
    db.session.add(new_user)
    db.session.commit()
    
    # Log activity
    activity = Activity(
        id=str(uuid.uuid4()),
        user_id=current_user.id,
        action='add_user',
        details=f'Added new user: {email}'
    )
    db.session.add(activity)
    db.session.commit()
    
    return jsonify({'message': 'User added successfully', 'user': new_user.to_dict()})

@app.route('/admin/bank/add', methods=['POST'])
@login_required
def add_bank():
    if not current_user.is_admin:
        return jsonify({'error': 'Access denied'}), 403
    
    data = request.json
    bank_name = data.get('name')
    account_number = data.get('accountNumber')
    account_type = data.get('accountType')
    user_id = data.get('userId')
    
    if not all([bank_name, account_number, account_type, user_id]):
        return jsonify({'error': 'All fields are required'}), 400
    
    # Check if user exists
    user = User.query.get(user_id)
    if not user:
        return jsonify({'error': 'User not found'}), 404
    
    # Create new bank
    new_bank = Bank(
        id=str(uuid.uuid4()),
        user_id=user_id,
        bank_name=bank_name,
        account_number=account_number,
        account_type=account_type
    )
    
    db.session.add(new_bank)
    db.session.commit()
    
    # Log activity
    activity = Activity(
        id=str(uuid.uuid4()),
        user_id=current_user.id,
        action='add_bank',
        details=f'Added new bank: {bank_name} for user: {user.email}'
    )
    db.session.add(activity)
    db.session.commit()
    
    return jsonify({'message': 'Bank added successfully', 'bank': new_bank.to_dict()})

@app.route('/admin/subscription/add', methods=['POST'])
@login_required
def add_subscription():
    if not current_user.is_admin:
        return jsonify({'error': 'Access denied'}), 403
    
    data = request.json
    user_id = data.get('user')
    plan = data.get('plan')
    start_date = data.get('startDate')
    end_date = data.get('endDate')
    
    if not all([user_id, plan, start_date, end_date]):
        return jsonify({'error': 'All fields are required'}), 400
    
    # Check if user exists
    user = User.query.get(user_id)
    if not user:
        return jsonify({'error': 'User not found'}), 404
    
    # Parse dates
    try:
        start_date = datetime.strptime(start_date, '%Y-%m-%d')
        end_date = datetime.strptime(end_date, '%Y-%m-%d')
    except ValueError:
        return jsonify({'error': 'Invalid date format'}), 400
    
    # Create new subscription
    new_subscription = Subscription(
        id=str(uuid.uuid4()),
        user_id=user_id,
        plan=plan,
        status='active',
        start_date=start_date,
        end_date=end_date
    )
    
    db.session.add(new_subscription)
    db.session.commit()
    
    # Log activity
    activity = Activity(
        id=str(uuid.uuid4()),
        user_id=current_user.id,
        action='add_subscription',
        details=f'Added new subscription: {plan} for user: {user.email}'
    )
    db.session.add(activity)
    db.session.commit()
    
    return jsonify({'message': 'Subscription added successfully', 'subscription': new_subscription.to_dict()})

# Admin routes for managing existing items
@app.route('/admin/bank/<string:bank_id>', methods=['PUT', 'DELETE'])
@login_required
def manage_bank(bank_id):
    if not current_user.is_admin:
        return jsonify({'error': 'Access denied'}), 403
    
    bank = Bank.query.get_or_404(bank_id)
    
    if request.method == 'DELETE':
        db.session.delete(bank)
        db.session.commit()
        
        # Log activity
        activity = Activity(
            id=str(uuid.uuid4()),
            user_id=current_user.id,
            action='delete_bank',
            details=f'Deleted bank: {bank.bank_name}'
        )
        db.session.add(activity)
        db.session.commit()
        
        return jsonify({'message': 'Bank deleted successfully'})
    
    data = request.json
    bank.bank_name = data.get('name', bank.bank_name)
    bank.account_number = data.get('account_number', bank.account_number)
    bank.account_type = data.get('account_type', bank.account_type)
    db.session.commit()
    
    # Log activity
    activity = Activity(
        id=str(uuid.uuid4()),
        user_id=current_user.id,
        action='update_bank',
        details=f'Updated bank: {bank.bank_name}'
    )
    db.session.add(activity)
    db.session.commit()
    
    return jsonify({'message': 'Bank updated successfully', 'bank': bank.to_dict()})

@app.route('/admin/user/<string:user_id>', methods=['PUT', 'DELETE'])
@login_required
def manage_user(user_id):
    if not current_user.is_admin:
        return jsonify({'error': 'Access denied'}), 403
    
    user = User.query.get_or_404(user_id)
    
    if request.method == 'DELETE':
        db.session.delete(user)
        db.session.commit()
        
        # Log activity
        activity = Activity(
            id=str(uuid.uuid4()),
            user_id=current_user.id,
            action='delete_user',
            details=f'Deleted user: {user.email}'
        )
        db.session.add(activity)
        db.session.commit()
        
        return jsonify({'message': 'User deleted successfully'})
    
    data = request.json
    user.name = data.get('name', user.name)
    user.email = data.get('email', user.email)
    user.role = data.get('role', user.role)
    user.active = data.get('active', user.active)
    db.session.commit()
    
    # Log activity
    activity = Activity(
        id=str(uuid.uuid4()),
        user_id=current_user.id,
        action='update_user',
        details=f'Updated user: {user.email}'
    )
    db.session.add(activity)
    db.session.commit()
    
    return jsonify({'message': 'User updated successfully', 'user': user.to_dict()})

@app.route('/admin/subscription/<string:subscription_id>', methods=['PUT', 'DELETE'])
@login_required
def manage_subscription(subscription_id):
    if not current_user.is_admin:
        return jsonify({'error': 'Access denied'}), 403
    
    subscription = Subscription.query.get_or_404(subscription_id)
    
    if request.method == 'DELETE':
        db.session.delete(subscription)
        db.session.commit()
        
        # Log activity
        activity = Activity(
            id=str(uuid.uuid4()),
            user_id=current_user.id,
            action='delete_subscription',
            details=f'Deleted subscription for user: {subscription.user.email}'
        )
        db.session.add(activity)
        db.session.commit()
        
        return jsonify({'message': 'Subscription deleted successfully'})
    
    data = request.json
    subscription.plan = data.get('plan', subscription.plan)
    subscription.start_date = data.get('start_date', subscription.start_date)
    subscription.end_date = data.get('end_date', subscription.end_date)
    subscription.status = data.get('status', subscription.status)
    db.session.commit()
    
    # Log activity
    activity = Activity(
        id=str(uuid.uuid4()),
        user_id=current_user.id,
        action='update_subscription',
        details=f'Updated subscription for user: {subscription.user.email}'
    )
    db.session.add(activity)
    db.session.commit()
    
    return jsonify({'message': 'Subscription updated successfully', 'subscription': subscription.to_dict()})

# Main execution block
if __name__ == '__main__':
    # For production, use a proper WSGI server like Gunicorn or Waitress
    # Example: gunicorn -w 4 'app:app'
    print("🚀 Starting Flask development server...")
    app.run(debug=True, port=5000) # Use port 5000 to match redirect URI
