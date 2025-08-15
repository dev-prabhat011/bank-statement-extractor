# 🚀 Bank Statement Extractor - Deployment Guide

## 📋 Prerequisites

- **Docker** (v20.10+)
- **Docker Compose** (v2.0+)
- **Git** (for version control)

## 🐳 Quick Start with Docker

### 1. Clone and Navigate
```bash
git clone <your-repo-url>
cd "Bank account extracter"
```

### 2. Deploy with One Command
```bash
chmod +x deploy.sh
./deploy.sh
```

### 3. Access Your Application
- **Local**: http://localhost:5000
- **Network**: http://your-server-ip:5000

## 🔧 Manual Deployment Steps

### Step 1: Build and Start
```bash
# Build the Docker image
docker-compose build

# Start the services
docker-compose up -d
```

### Step 2: Verify Deployment
```bash
# Check if containers are running
docker-compose ps

# View logs
docker-compose logs -f

# Test health endpoint
curl http://localhost:5000/health
```

### Step 3: Access the Application
Open your browser and navigate to `http://localhost:5000`

## 🌐 Production Deployment

### Option 1: VPS/Cloud Server
1. **Upload your code** to your server
2. **Install Docker**: `curl -fsSL https://get.docker.com | sh`
3. **Install Docker Compose**: `sudo apt install docker-compose`
4. **Run deployment script**: `./deploy.sh`

### Option 2: Heroku
```bash
# Install Heroku CLI
# Create Procfile
echo "web: gunicorn app:app" > Procfile

# Deploy
heroku create your-app-name
git add .
git commit -m "Initial deployment"
git push heroku main
```

### Option 3: AWS/GCP/Azure
- **AWS**: Deploy to EC2 or use Elastic Beanstalk
- **GCP**: Use App Engine or Cloud Run
- **Azure**: Use App Service or Container Instances

## 📁 Directory Structure
```
Bank account extracter/
├── app.py                 # Flask application
├── Dockerfile            # Docker configuration
├── docker-compose.yml    # Docker services
├── requirements.txt      # Python dependencies
├── deploy.sh            # Deployment script
├── uploads/             # PDF upload directory
├── outputs/             # Generated files
└── logs/                # Application logs
```

## 🔒 Security Considerations

### Environment Variables
Create a `.env` file:
```bash
FLASK_SECRET_KEY=your-super-secret-key-here
FLASK_ENV=production
MAX_CONTENT_LENGTH=16777216
```

### SSL/HTTPS
For production, add SSL certificates:
```bash
# Generate self-signed certificate (development)
openssl req -x509 -newkey rsa:4096 -keyout ssl/key.pem -out ssl/cert.pem -days 365 -nodes

# Use Let's Encrypt for production
certbot certonly --standalone -d yourdomain.com
```

## 📊 Monitoring and Maintenance

### View Logs
```bash
# Application logs
docker-compose logs -f bank-extractor

# Nginx logs (if using)
docker-compose logs -f nginx
```

### Update Application
```bash
# Pull latest changes
git pull origin main

# Rebuild and restart
docker-compose down
docker-compose up --build -d
```

### Backup Data
```bash
# Backup uploads and outputs
tar -czf backup-$(date +%Y%m%d).tar.gz uploads/ outputs/

# Backup database
docker-compose exec bank-extractor sqlite3 app.db ".backup backup.db"
```

## 🚨 Troubleshooting

### Common Issues

1. **Port already in use**
   ```bash
   # Check what's using port 5000
   lsof -i :5000
   
   # Kill the process or change port in docker-compose.yml
   ```

2. **Java not found (tabula-py error)**
   ```bash
   # Rebuild Docker image
   docker-compose build --no-cache
   ```

3. **Permission denied**
   ```bash
   # Fix directory permissions
   sudo chown -R $USER:$USER uploads outputs logs
   chmod 755 uploads outputs logs
   ```

### Health Checks
```bash
# Check application health
curl http://localhost:5000/health

# Check Docker containers
docker-compose ps

# Check resource usage
docker stats
```

## 📈 Scaling

### Horizontal Scaling
```bash
# Scale to multiple instances
docker-compose up -d --scale bank-extractor=3
```

### Load Balancer
Use Nginx as reverse proxy (already configured in docker-compose.yml):
```bash
# Start with Nginx
docker-compose up -d nginx
```

## 🔄 CI/CD Pipeline

### GitHub Actions Example
Create `.github/workflows/deploy.yml`:
```yaml
name: Deploy to Production
on:
  push:
    branches: [main]
jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Deploy to server
        run: |
          # Your deployment commands here
```

## 📞 Support

- **Documentation**: Check the main README.md
- **Issues**: Create an issue in your repository
- **Logs**: Check `logs/` directory for detailed error information

---

**Happy Deploying! 🎉**
