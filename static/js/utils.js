// static/js/utils.js - Common utility functions for Bank Account Extractor

class BankExtractorUtils {
    constructor() {
        this.init();
    }

    init() {
        this.setupFormValidation();
        this.setupAccessibility();
        this.setupNotifications();
        this.setupFileUpload();
    }

    // Form validation utilities
    setupFormValidation() {
        // Add real-time validation to forms
        document.querySelectorAll('form').forEach(form => {
            form.addEventListener('input', (e) => {
                this.validateField(e.target);
            });

            form.addEventListener('submit', (e) => {
                if (!this.validateForm(form)) {
                    e.preventDefault();
                    this.showNotification('Please fix the errors before submitting.', 'error');
                }
            });
        });
    }

    validateField(field) {
        const value = field.value.trim();
        const fieldName = field.name || field.id || 'Field';
        let isValid = true;
        let errorMessage = '';

        // Remove existing error styling
        field.classList.remove('is-invalid');
        this.removeFieldError(field);

        // Required field validation
        if (field.hasAttribute('required') && !value) {
            isValid = false;
            errorMessage = `${fieldName} is required.`;
        }

        // Email validation
        if (field.type === 'email' && value && !this.isValidEmail(value)) {
            isValid = false;
            errorMessage = 'Please enter a valid email address.';
        }

        // PDF Password validation (can be simple or complex)
        if (field.type === 'password' && value && !this.isValidPdfPassword(value)) {
            isValid = false;
            errorMessage = 'PDF password can be any characters. Leave blank if not password protected.';
        }

        // File validation
        if (field.type === 'file' && field.files.length > 0) {
            const file = field.files[0];
            if (!this.isValidFile(file)) {
                isValid = false;
                errorMessage = 'Please select a valid PDF file (max 16MB).';
            }
        }

        if (!isValid) {
            field.classList.add('is-invalid');
            this.showFieldError(field, errorMessage);
        }

        return isValid;
    }

    validateForm(form) {
        let isValid = true;
        const fields = form.querySelectorAll('input, select, textarea');
        
        fields.forEach(field => {
            if (!this.validateField(field)) {
                isValid = false;
            }
        });

        return isValid;
    }

    isValidEmail(email) {
        const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
        return emailRegex.test(email);
    }

    isValidPdfPassword(password) {
        // PDF passwords can be any characters, including simple ones
        // Empty password is valid (for unprotected PDFs)
        // Any non-empty password is valid
        return true; // Always valid since PDF passwords can be anything
    }

    isValidFile(file) {
        const allowedTypes = ['application/pdf'];
        const maxSize = 16 * 1024 * 1024; // 16MB
        
        return allowedTypes.includes(file.type) && file.size <= maxSize;
    }

    showFieldError(field, message) {
        let errorDiv = field.parentNode.querySelector('.invalid-feedback');
        if (!errorDiv) {
            errorDiv = document.createElement('div');
            errorDiv.className = 'invalid-feedback';
            field.parentNode.appendChild(errorDiv);
        }
        errorDiv.textContent = message;
    }

    removeFieldError(field) {
        const errorDiv = field.parentNode.querySelector('.invalid-feedback');
        if (errorDiv) {
            errorDiv.remove();
        }
    }

    // Accessibility utilities
    setupAccessibility() {
        this.addSkipLink();
        this.addFocusIndicators();
        this.addKeyboardNavigation();
    }

    addSkipLink() {
        const skipLink = document.createElement('a');
        skipLink.href = '#main-content';
        skipLink.className = 'sr-only sr-only-focusable position-absolute top-0 start-0 p-3 bg-dark text-white text-decoration-none';
        skipLink.textContent = 'Skip to main content';
        document.body.insertBefore(skipLink, document.body.firstChild);
    }

    addFocusIndicators() {
        document.addEventListener('focusin', (e) => {
            e.target.classList.add('focus-visible');
        });
        
        document.addEventListener('focusout', (e) => {
            e.target.classList.remove('focus-visible');
        });
    }

    addKeyboardNavigation() {
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Tab') {
                document.body.classList.add('keyboard-navigation');
            }
        });
        
        document.addEventListener('mousedown', () => {
            document.body.classList.remove('keyboard-navigation');
        });
    }

    // Notification system
    setupNotifications() {
        // Create notification container if it doesn't exist
        if (!document.getElementById('notification-container')) {
            const container = document.createElement('div');
            container.id = 'notification-container';
            container.className = 'position-fixed top-0 end-0 p-3';
            container.style.zIndex = '9999';
            document.body.appendChild(container);
        }
    }

    showNotification(message, type = 'info', duration = 5000) {
        const container = document.getElementById('notification-container');
        const toast = document.createElement('div');
        
        const bgClass = {
            'success': 'bg-success',
            'error': 'bg-danger',
            'warning': 'bg-warning',
            'info': 'bg-info'
        }[type] || 'bg-info';
        
        toast.className = `toast align-items-center text-white border-0 ${bgClass}`;
        toast.innerHTML = `
            <div class="d-flex">
                <div class="toast-body">${message}</div>
                <button type="button" class="btn-close btn-close-white me-2 m-auto" data-bs-dismiss="toast"></button>
            </div>
        `;
        
        container.appendChild(toast);
        
        const bsToast = new bootstrap.Toast(toast);
        bsToast.show();
        
        setTimeout(() => {
            if (toast.parentNode) {
                toast.parentNode.removeChild(toast);
            }
        }, duration);
    }

    // File upload utilities
    setupFileUpload() {
        document.querySelectorAll('input[type="file"]').forEach(input => {
            input.addEventListener('change', (e) => {
                this.handleFileSelect(e.target);
            });
        });
    }

    handleFileSelect(input) {
        if (input.files.length > 0) {
            const file = input.files[0];
            this.showFileInfo(input, file);
        } else {
            this.hideFileInfo(input);
        }
    }

    showFileInfo(input, file) {
        let infoDiv = input.parentNode.querySelector('.file-info');
        if (!infoDiv) {
            infoDiv = document.createElement('div');
            infoDiv.className = 'file-info mt-2 p-2 bg-light rounded';
            input.parentNode.appendChild(infoDiv);
        }
        
        const size = (file.size / (1024 * 1024)).toFixed(2);
        infoDiv.innerHTML = `
            <small class="text-muted">
                <i class="bi bi-file-pdf me-1"></i>
                ${file.name} (${size} MB)
            </small>
        `;
    }

    hideFileInfo(input) {
        const infoDiv = input.parentNode.querySelector('.file-info');
        if (infoDiv) {
            infoDiv.remove();
        }
    }

    // Utility methods
    debounce(func, wait) {
        let timeout;
        return function executedFunction(...args) {
            const later = () => {
                clearTimeout(timeout);
                func(...args);
            };
            clearTimeout(timeout);
            timeout = setTimeout(later, wait);
        };
    }

    throttle(func, limit) {
        let inThrottle;
        return function() {
            const args = arguments;
            const context = this;
            if (!inThrottle) {
                func.apply(context, args);
                inThrottle = true;
                setTimeout(() => inThrottle = false, limit);
            }
        };
    }

    // Date utilities
    formatDate(date) {
        return new Intl.DateTimeFormat('en-US', {
            year: 'numeric',
            month: 'short',
            day: 'numeric'
        }).format(new Date(date));
    }

    formatCurrency(amount, currency = 'INR') {
        return new Intl.NumberFormat('en-IN', {
            style: 'currency',
            currency: currency
        }).format(amount);
    }

    // API utilities
    async apiCall(url, options = {}) {
        try {
            const response = await fetch(url, {
                headers: {
                    'Content-Type': 'application/json',
                    ...options.headers
                },
                ...options
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            return await response.json();
        } catch (error) {
            console.error('API call failed:', error);
            this.showNotification('An error occurred. Please try again.', 'error');
            throw error;
        }
    }
}

// Initialize utilities when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.bankExtractorUtils = new BankExtractorUtils();
});

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = BankExtractorUtils;
}
