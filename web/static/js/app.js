/**
 * NeuronMap Web Interface JavaScript
 * Enhanced application logic and utilities
 */

// Global variables
window.neuronMap = {
    currentAnalysisId: null,
    analysisInterval: null,
    apiBaseUrl: '/api',
    config: {
        pollInterval: 1000,
        maxRetries: 3,
        timeout: 30000,
        dashboardRefreshInterval: 10000
    },
    state: {
        isOnline: true,
        lastUpdate: null,
        activeJobs: new Map()
    }
};

// Enhanced utility functions
const utils = {
    /**
     * Show loading spinner on button
     */
    showButtonLoading: function(button, text = 'Loading...') {
        const originalText = button.innerHTML;
        button.setAttribute('data-original-text', originalText);
        button.innerHTML = `<span class="spinner-border spinner-border-sm me-2" role="status"></span>${text}`;
        button.disabled = true;
    },

    /**
     * Hide loading spinner on button
     */
    hideButtonLoading: function(button) {
        const originalText = button.getAttribute('data-original-text');
        if (originalText) {
            button.innerHTML = originalText;
            button.removeAttribute('data-original-text');
        }
        button.disabled = false;
    },

    /**
     * Enhanced toast notification with icons
     */
    showToast: function(message, type = 'info', duration = 5000) {
        const icons = {
            'success': 'fas fa-check-circle',
            'error': 'fas fa-exclamation-triangle',
            'warning': 'fas fa-exclamation-circle',
            'info': 'fas fa-info-circle'
        };

        const toast = document.createElement('div');
        toast.className = `toast align-items-center text-white bg-${type === 'error' ? 'danger' : type} border-0`;
        toast.setAttribute('role', 'alert');
        toast.innerHTML = `
            <div class="d-flex">
                <div class="toast-body">
                    <i class="${icons[type] || icons.info} me-2"></i>${message}
                </div>
                <button type="button" class="btn-close btn-close-white me-2 m-auto" data-bs-dismiss="toast"></button>
            </div>
        `;

        // Create toast container if it doesn't exist
        let container = document.getElementById('toast-container');
        if (!container) {
            container = document.createElement('div');
            container.id = 'toast-container';
            container.className = 'toast-container position-fixed top-0 end-0 p-3';
            container.style.zIndex = '1060';
            document.body.appendChild(container);
        }

        container.appendChild(toast);
        const bsToast = new bootstrap.Toast(toast, {
            autohide: duration > 0,
            delay: duration
        });
        bsToast.show();

        // Remove from DOM after hiding
        toast.addEventListener('hidden.bs.toast', () => {
            toast.remove();
        });

        return bsToast;
    },

    /**
     * Show loading state with skeleton
     */
    showSkeleton: function(element, rows = 3) {
        const skeleton = Array(rows).fill(0).map(() => 
            '<div class="loading-shimmer rounded mb-2" style="height: 20px;"></div>'
        ).join('');
        element.innerHTML = skeleton;
    },

    /**
     * API request wrapper with error handling
     */
    apiRequest: async function(endpoint, options = {}) {
        const url = `${window.neuronMap.apiBaseUrl}${endpoint}`;
        const defaultOptions = {
            headers: {
                'Content-Type': 'application/json',
                'X-Requested-With': 'XMLHttpRequest'
            },
            timeout: window.neuronMap.config.timeout
        };

        try {
            const response = await fetch(url, { ...defaultOptions, ...options });
            
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }

            const data = await response.json();
            window.neuronMap.state.isOnline = true;
            return data;
            
        } catch (error) {
            window.neuronMap.state.isOnline = false;
            console.error(`API request failed for ${endpoint}:`, error);
            throw error;
        }
    },

    /**
     * Update system status indicator
     */
    updateSystemStatus: function(isOnline = null) {
        if (isOnline !== null) {
            window.neuronMap.state.isOnline = isOnline;
        }

        const statusBadge = document.getElementById('status-badge');
        if (statusBadge) {
            if (window.neuronMap.state.isOnline) {
                statusBadge.className = 'badge bg-success ms-2';
                statusBadge.textContent = 'Online';
            } else {
                statusBadge.className = 'badge bg-danger ms-2';
                statusBadge.textContent = 'Offline';
            }
        }
    },

    /**
     * Format file size
     */
    formatFileSize: function(bytes) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    },

    /**
     * Format duration
     */
    formatDuration: function(seconds) {
        const hours = Math.floor(seconds / 3600);
        const minutes = Math.floor((seconds % 3600) / 60);
        const secs = Math.floor(seconds % 60);
        
        if (hours > 0) {
            return `${hours}h ${minutes}m ${secs}s`;
        } else if (minutes > 0) {
            return `${minutes}m ${secs}s`;
        } else {
            return `${secs}s`;
        }
    },

    /**
     * Debounce function
     */
    debounce: function(func, delay) {
        let timeoutId;
        return function (...args) {
            clearTimeout(timeoutId);
            timeoutId = setTimeout(() => func.apply(this, args), delay);
        };
    },

    /**
     * Copy text to clipboard
     */
    copyToClipboard: function(text) {
        if (navigator.clipboard) {
            navigator.clipboard.writeText(text).then(() => {
                utils.showToast('Copied to clipboard!', 'success', 2000);
            }).catch(() => {
                utils.showToast('Failed to copy to clipboard', 'error');
            });
        } else {
            // Fallback for older browsers
            const textArea = document.createElement('textarea');
            textArea.value = text;
            document.body.appendChild(textArea);
            textArea.select();
            try {
                document.execCommand('copy');
                utils.showToast('Copied to clipboard!', 'success', 2000);
            } catch (err) {
                utils.showToast('Failed to copy to clipboard', 'error');
            }
            document.body.removeChild(textArea);
        }
    },

    /**
     * Download file from blob
     */
    downloadFile: function(blob, filename) {
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = filename;
        document.body.appendChild(a);
        a.click();
        window.URL.revokeObjectURL(url);
        document.body.removeChild(a);
    },

    /**
     * Validate form data
     */
    validateForm: function(formElement) {
        const errors = [];
        const inputs = formElement.querySelectorAll('input[required], select[required], textarea[required]');
        
        inputs.forEach(input => {
            if (!input.value.trim()) {
                errors.push(`${input.getAttribute('placeholder') || input.name || 'Field'} is required`);
                input.classList.add('is-invalid');
            } else {
                input.classList.remove('is-invalid');
            }
        });

        return errors;
    }
};

// Enhanced dashboard functionality
const dashboard = {
    /**
     * Initialize dashboard
     */
    init: function() {
        this.loadStats();
        this.loadRecentActivity();
        this.loadAvailableModels();
        
        // Set up periodic refresh
        setInterval(() => {
            this.loadStats();
            this.loadRecentActivity();
        }, window.neuronMap.config.dashboardRefreshInterval);

        // Set up page visibility API to pause/resume updates
        document.addEventListener('visibilitychange', () => {
            if (document.visibilityState === 'visible') {
                this.loadStats();
                this.loadRecentActivity();
            }
        });
    },

    /**
     * Load system statistics
     */
    loadStats: async function() {
        try {
            const data = await utils.apiRequest('/stats');
            
            // Animate number updates
            this.animateNumber('total-analyses', data.total_analyses || 0);
            this.animateNumber('models-analyzed', data.models_analyzed || 0);
            this.animateNumber('layers-processed', data.layers_processed || 0);
            this.animateNumber('visualizations-created', data.visualizations_created || 0);
            
            utils.updateSystemStatus(true);
            
        } catch (error) {
            console.error('Error loading stats:', error);
            utils.updateSystemStatus(false);
        }
    },

    /**
     * Animate number change
     */
    animateNumber: function(elementId, targetValue) {
        const element = document.getElementById(elementId);
        if (!element) return;

        const currentValue = parseInt(element.textContent) || 0;
        const diff = targetValue - currentValue;
        const steps = 20;
        const stepValue = diff / steps;
        let step = 0;

        const animate = () => {
            if (step < steps) {
                element.textContent = Math.round(currentValue + (stepValue * step));
                step++;
                requestAnimationFrame(animate);
            } else {
                element.textContent = targetValue;
            }
        };

        if (diff !== 0) {
            animate();
        }
    },

    /**
     * Load recent activity
     */
    loadRecentActivity: async function() {
        try {
            const data = await utils.apiRequest('/recent-activity');
            const container = document.getElementById('recent-activities');
            
            if (!container) return;

            if (data.activities && data.activities.length > 0) {
                container.innerHTML = data.activities.map(activity => `
                    <div class="border-bottom p-2 activity-item">
                        <div class="d-flex align-items-center">
                            <i class="fas fa-${activity.icon} text-${activity.color} me-2"></i>
                            <span class="flex-grow-1">${activity.message}</span>
                            <small class="text-muted">${activity.timestamp}</small>
                        </div>
                    </div>
                `).join('');
            } else {
                container.innerHTML = '<div class="text-center p-3 text-muted">No recent activities</div>';
            }
            
            // Update timestamp
            const lastUpdated = document.getElementById('last-updated');
            if (lastUpdated) {
                lastUpdated.textContent = new Date().toLocaleTimeString();
            }
            
        } catch (error) {
            console.error('Error loading activities:', error);
            const container = document.getElementById('recent-activities');
            if (container) {
                container.innerHTML = '<div class="text-center p-3 text-danger"><i class="fas fa-exclamation-triangle me-2"></i>Error loading activities</div>';
            }
        }
    },

    /**
     * Load available models
     */
    loadAvailableModels: async function() {
        try {
            const data = await utils.apiRequest('/models');
            const container = document.getElementById('model-list');
            
            if (!container) return;

            if (data.models && data.models.length > 0) {
                container.innerHTML = data.models.slice(0, 6).map(model => `
                    <div class="col-md-4 col-lg-2 mb-3">
                        <div class="card border h-100 model-card" onclick="dashboard.analyzeModel('${model}')">
                            <div class="card-body text-center p-2">
                                <i class="fas fa-robot text-primary mb-2"></i>
                                <h6 class="card-title small mb-0" title="${model}">${this.truncateText(model, 12)}</h6>
                            </div>
                        </div>
                    </div>
                `).join('') + (data.models.length > 6 ? 
                    `<div class="col-12 text-center">
                        <small class="text-muted">... and ${data.models.length - 6} more models</small>
                        <br><a href="/analysis" class="btn btn-sm btn-outline-primary mt-2">View All Models</a>
                    </div>` : '');
            } else {
                container.innerHTML = '<div class="col-12 text-center text-muted">No models available</div>';
            }
            
        } catch (error) {
            console.error('Error loading models:', error);
            const container = document.getElementById('model-list');
            if (container) {
                container.innerHTML = '<div class="col-12 text-center text-danger"><i class="fas fa-exclamation-triangle me-2"></i>Error loading models</div>';
            }
        }
    },

    /**
     * Truncate text for display
     */
    truncateText: function(text, maxLength) {
        return text.length > maxLength ? text.substring(0, maxLength) + '...' : text;
    },

    /**
     * Quick analysis with selected model
     */
    analyzeModel: function(modelName) {
        window.location.href = `/analysis?model=${encodeURIComponent(modelName)}`;
    }
};

// Enhanced analysis functionality
const analysis = {
    /**
     * Start analysis job
     */
    start: async function(formData) {
        try {
            const data = await utils.apiRequest('/analysis/start', {
                method: 'POST',
                body: formData
            });

            if (data.analysis_id) {
                window.neuronMap.currentAnalysisId = data.analysis_id;
                utils.showToast('Analysis started successfully!', 'success');
                this.startProgressMonitoring(data.analysis_id);
                return data.analysis_id;
            } else {
                throw new Error('No analysis ID returned');
            }
            
        } catch (error) {
            console.error('Error starting analysis:', error);
            utils.showToast('Failed to start analysis: ' + error.message, 'error');
            throw error;
        }
    },

    /**
     * Monitor analysis progress
     */
    startProgressMonitoring: function(analysisId) {
        // Clear any existing interval
        if (window.neuronMap.analysisInterval) {
            clearInterval(window.neuronMap.analysisInterval);
        }

        // Start monitoring
        window.neuronMap.analysisInterval = setInterval(async () => {
            try {
                const status = await this.getStatus(analysisId);
                this.updateProgress(status);

                if (status.status === 'completed' || status.status === 'failed') {
                    clearInterval(window.neuronMap.analysisInterval);
                    window.neuronMap.analysisInterval = null;
                }
            } catch (error) {
                console.error('Error monitoring progress:', error);
            }
        }, 1000);
    },

    /**
     * Get analysis status
     */
    getStatus: async function(analysisId) {
        return await utils.apiRequest(`/analysis/status/${analysisId}`);
    },

    /**
     * Update progress display
     */
    updateProgress: function(status) {
        const progressBar = document.getElementById('analysis-progress');
        const statusText = document.getElementById('analysis-status');
        const progressText = document.getElementById('progress-text');

        if (progressBar) {
            progressBar.style.width = `${status.progress || 0}%`;
            progressBar.setAttribute('aria-valuenow', status.progress || 0);
        }

        if (statusText) {
            statusText.textContent = status.message || 'Processing...';
        }

        if (progressText) {
            progressText.textContent = `${status.progress || 0}%`;
        }

        // Update status icon and color
        if (status.status === 'completed') {
            if (progressBar) {
                progressBar.className = 'progress-bar bg-success';
            }
            utils.showToast('Analysis completed successfully!', 'success');
        } else if (status.status === 'failed') {
            if (progressBar) {
                progressBar.className = 'progress-bar bg-danger';
            }
            utils.showToast('Analysis failed: ' + (status.error || 'Unknown error'), 'error');
        }
    }
};

// Initialize when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    // Initialize dashboard if on main page
    if (document.getElementById('total-analyses')) {
        dashboard.init();
    }

    // Initialize tooltips
    const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    tooltipTriggerList.map(function(tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });

    // Initialize popovers
    const popoverTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="popover"]'));
    popoverTriggerList.map(function(popoverTriggerEl) {
        return new bootstrap.Popover(popoverTriggerEl);
    });

    // Handle page visibility changes
    document.addEventListener('visibilitychange', () => {
        if (document.visibilityState === 'visible' && window.neuronMap.currentAnalysisId) {
            // Resume monitoring if there's an active analysis
            analysis.startProgressMonitoring(window.neuronMap.currentAnalysisId);
        }
    });
});

// Global functions for template usage
window.quickAnalysis = function() {
    window.location.href = '/analysis?demo=true';
};

window.showSystemInfo = function() {
    const info = `
        <div class="modal fade" id="systemInfoModal" tabindex="-1">
            <div class="modal-dialog">
                <div class="modal-content">
                    <div class="modal-header">
                        <h5 class="modal-title"><i class="fas fa-info-circle me-2"></i>System Information</h5>
                        <button type="button" class="btn-close" data-bs-dismiss="modal"></button>
                    </div>
                    <div class="modal-body">
                        <table class="table table-sm">
                            <tr><td><strong>Platform:</strong></td><td>NeuronMap v2.0</td></tr>
                            <tr><td><strong>Interface:</strong></td><td>Flask Web Application</td></tr>
                            <tr><td><strong>Features:</strong></td><td>Advanced Analytics, Multi-Model Support</td></tr>
                            <tr><td><strong>Status:</strong></td><td><span class="badge bg-success">Online</span></td></tr>
                            <tr><td><strong>Last Updated:</strong></td><td>${new Date().toLocaleString()}</td></tr>
                        </table>
                    </div>
                    <div class="modal-footer">
                        <button type="button" class="btn btn-secondary" data-bs-dismiss="modal">Close</button>
                    </div>
                </div>
            </div>
        </div>
    `;
    
    // Remove existing modal
    const existingModal = document.getElementById('systemInfoModal');
    if (existingModal) existingModal.remove();
    
    // Add new modal
    document.body.insertAdjacentHTML('beforeend', info);
    const modal = new bootstrap.Modal(document.getElementById('systemInfoModal'));
    modal.show();
};

window.refreshModels = function() {
    if (dashboard.loadAvailableModels) {
        dashboard.loadAvailableModels();
        utils.showToast('Model list refreshed', 'info', 2000);
    }
};
