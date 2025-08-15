import os
import requests
from pathlib import Path

# Create images directory if it doesn't exist
images_dir = Path('static/images')
images_dir.mkdir(parents=True, exist_ok=True)

# Image URLs
images = {
    'hero-image.jpg': 'https://images.unsplash.com/photo-1551288049-bebda4e38f71?w=800&auto=format&fit=crop',
    'python-logo.png': 'https://www.python.org/static/img/python-logo.png',
    'flask-logo.png': 'https://cdn.buttercms.com/w8lc0UqsQCnPG0AjyfK5',
    'security-badge.png': 'https://cdn-icons-png.flaticon.com/512/2889/2889676.png'
}

def download_image(url, filename):
    try:
        response = requests.get(url)
        response.raise_for_status()
        
        file_path = images_dir / filename
        with open(file_path, 'wb') as f:
            f.write(response.content)
        print(f"Successfully downloaded {filename}")
    except Exception as e:
        print(f"Error downloading {filename}: {str(e)}")

def main():
    print("Starting image downloads...")
    for filename, url in images.items():
        download_image(url, filename)
    print("Image download process completed!")

if __name__ == "__main__":
    main() 