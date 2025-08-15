@echo off
echo 🚀 Bank Statement Extractor - Deployment Script
echo ================================================

REM Check if Docker is installed
docker --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Docker is not installed. Please install Docker Desktop first.
    echo    Download from: https://www.docker.com/products/docker-desktop/
    pause
    exit /b 1
)

REM Check if Docker Compose is installed
docker-compose --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Docker Compose is not installed. Please install Docker Compose first.
    pause
    exit /b 1
)

echo ✅ Docker and Docker Compose are available

REM Create necessary directories
echo 📁 Creating necessary directories...
if not exist "uploads" mkdir uploads
if not exist "outputs" mkdir outputs
if not exist "logs" mkdir logs
if not exist "ssl" mkdir ssl

REM Build and start the application
echo 🔨 Building and starting the application...
docker-compose up --build -d

REM Wait for the application to start
echo ⏳ Waiting for the application to start...
timeout /t 10 /nobreak >nul

REM Check if the application is running
echo 🔍 Checking application status...
curl -f http://localhost:5000/health >nul 2>&1
if %errorlevel% equ 0 (
    echo ✅ Application is running successfully!
    echo.
    echo 🌐 Access your application at:
    echo    Local: http://localhost:5000
    echo    Network: http://localhost:5000
    echo.
    echo 📁 Uploads directory: .\uploads
    echo 📁 Outputs directory: .\outputs
    echo 📁 Logs directory: .\logs
    echo.
    echo 🔧 Management commands:
    echo    View logs: docker-compose logs -f
    echo    Stop: docker-compose down
    echo    Restart: docker-compose restart
    echo    Update: docker-compose pull && docker-compose up -d
) else (
    echo ❌ Application failed to start. Check logs with:
    echo    docker-compose logs
    pause
    exit /b 1
)

echo.
echo 🎉 Deployment completed successfully!
pause
