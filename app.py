import os
import json
from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from main import (
    save_campaign_data, update_progress, load_campaign_data, load_progress_data,
    extract_contacts, generate_emails_background, send_emails_background,
    get_common_keys, EmailConfig, CampaignStatus
)
from app_routes import api_routes

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'data'
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max upload size
app.config['TEMPLATES_AUTO_RELOAD'] = True
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0  # Disable caching for static files

# Register blueprints
app.register_blueprint(api_routes, url_prefix='/api')

# Ensure data directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

@app.route('/')
def index():
    # Get all campaigns using the /analytics/overall endpoint
    try:
        from main import overall_analytics
        # Call the API function directly instead of making an HTTP request
        response = overall_analytics()
        # The response is already a Flask response object, we need to extract the data
        if response.status_code == 200:
            analytics_data = response.json
            
            # Extract campaign data from the analytics response
            campaign_details = analytics_data.get('campaign_breakdown', [])
            campaigns = []
            
            # Transform campaign data to match the expected format for the template
            for campaign in campaign_details:
                campaign_id = campaign.get('campaign_id', '')
                
                # Format campaign data for the template
                campaign_info = {
                    'id': campaign_id,
                    'status': campaign.get('status', 'unknown'),
                    'total_apps': campaign.get('total_apps', 0),
                    'apps_with_contacts': campaign.get('apps_with_contacts', 0),
                    'initial_emails_sent': campaign.get('initial_emails_sent', 0),
                    'reply_rate': campaign.get('reply_rate', 0),
                    'stats_summary': campaign
                }
                campaigns.append(campaign_info)
            
            # Extract overall stats
            overall_summary = analytics_data.get('overall_summary', {})
            growth_data = overall_summary.get('growth', {})
            
            # Map analytics data to the stats format expected by the template
            stats = {
                'total_campaigns': overall_summary.get('total_campaigns', 0),
                'total_emails_sent': overall_summary.get('initial_emails_sent', 0),
                'reply_rate': overall_summary.get('overall_reply_rate', 0),
                'total_apps': overall_summary.get('total_apps', 0),
                # Use growth values from the enhanced analytics endpoint
                'campaign_growth': growth_data.get('campaign_growth', 0),
                'email_growth': growth_data.get('email_growth', 0),
                'reply_rate_growth': growth_data.get('reply_rate_growth', 0),
                'app_growth': growth_data.get('app_growth', 0)
            }
        else:
            # Handle error response
            campaigns = []
            stats = {
                'total_campaigns': 0,
                'total_emails_sent': 0,
                'reply_rate': 0,
                'total_apps': 0,
                'campaign_growth': 0,
                'email_growth': 0,
                'reply_rate_growth': 0,
                'app_growth': 0
            }
        
        return render_template('new_dashboard.html', campaigns=campaigns, stats=stats)
    except Exception as e:
        # If there's an error, still render the template but with empty data
        return render_template('new_dashboard.html', campaigns=[], stats={
            'total_campaigns': 0,
            'total_emails_sent': 0,
            'reply_rate': 0,
            'total_apps': 0,
            'campaign_growth': 0,
            'email_growth': 0,
            'reply_rate_growth': 0,
            'app_growth': 0
        }, error=str(e))

@app.route('/analytics')
def analytics():
    return render_template('analytics.html')

@app.route('/settings', methods=['GET', 'POST'])
def settings():
    success_message = None
    error_message = None
    
    # Load current settings from .env file
    env_path = os.path.join(os.path.dirname(__file__), '.env')
    current_settings = {}
    
    try:
        with open(env_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    current_settings[key] = value
    except Exception as e:
        error_message = f"Error loading settings: {str(e)}"
    
    # Handle form submission
    if request.method == 'POST':
        try:
            # Get form data
            new_settings = {
                'OPENAI_API_KEY': request.form.get('openai_api_key', ''),
                'OPENAI_MODEL': request.form.get('openai_model', 'gpt-4o'),
                'SYSTEM_PROMPT': request.form.get('system_prompt', ''),
                'GMAIL_USER': request.form.get('gmail_user', ''),
                'GMAIL_PASSWORD': request.form.get('gmail_password', ''),
                'TEST_EMAIL': request.form.get('test_email', ''),
                'EMAIL_SENDER_NAME': request.form.get('email_sender_name', 'UndrApp Intel'),
                'EMAIL_REPLY_TO': request.form.get('email_reply_to', '')
            }
            
            # Update .env file
            env_content = []
            with open(env_path, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    if line.strip() and not line.startswith('#') and '=' in line:
                        key = line.split('=', 1)[0].strip()
                        if key in new_settings:
                            env_content.append(f"{key}={new_settings[key]}\n")
                            del new_settings[key]
                        else:
                            env_content.append(line)
                    else:
                        env_content.append(line)
            
            # Add any new settings that weren't in the original file
            for key, value in new_settings.items():
                if value:  # Only add non-empty values
                    env_content.append(f"{key}={value}\n")
            
            with open(env_path, 'w') as f:
                f.writelines(env_content)
            
            # Import and call the reload_env_vars function from main.py
            from main import reload_env_vars
            reload_env_vars()
            
            success_message = "Settings updated successfully and applied immediately."
            
            # Reload current settings
            current_settings = {}
            with open(env_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        key, value = line.split('=', 1)
                        current_settings[key] = value
                        
        except Exception as e:
            error_message = f"Error saving settings: {str(e)}"
    
    # Prepare settings for template
    template_settings = {
        'openai_api_key': current_settings.get('OPENAI_API_KEY', ''),
        'openai_model': current_settings.get('OPENAI_MODEL', 'gpt-4o'),
        'system_prompt': current_settings.get('SYSTEM_PROMPT', ''),
        'gmail_user': current_settings.get('GMAIL_USER', ''),
        'gmail_password': current_settings.get('GMAIL_PASSWORD', ''),
        'test_email': current_settings.get('TEST_EMAIL', ''),
        'email_sender_name': current_settings.get('EMAIL_SENDER_NAME', 'UndrApp Intel'),
        'email_reply_to': current_settings.get('EMAIL_REPLY_TO', current_settings.get('GMAIL_USER', ''))
    }
    
    return render_template('settings.html', settings=template_settings, success_message=success_message, error_message=error_message)

@app.route('/campaign/new')
def new_campaign():
    return render_template('new_campaign.html')

@app.route('/campaign/<campaign_id>')
def campaign_details(campaign_id):
    # Load campaign data
    try:
        from main import calculate_campaign_stats
        campaign_data = load_campaign_data(campaign_id)
        progress_data = load_progress_data(campaign_id)
        
        if campaign_data and progress_data:
            # Calculate stats
            stats = calculate_campaign_stats(campaign_data)
            
            # Calculate percentages for progress bars
            extraction_percentage = 100 if progress_data.get('stage') != 'uploaded' else 0
            generation_percentage = 100 if progress_data.get('stage') in ['emails_generated', 'email_sending', 'followups_active', 'completed'] else 0
            sending_percentage = 100 if progress_data.get('stage') in ['followups_active', 'completed'] else (50 if progress_data.get('stage') == 'email_sending' else 0)
            
            # Get email stats
            total_emails_sent = stats.get('initial_emails_sent', 0) + stats.get('total_followups_sent', 0)
            total_replies = stats.get('total_replies', 0)
            reply_rate = stats.get('reply_rate', 0)
            
            # Combine data
            campaign = {
                'id': campaign_id,
                'status': progress_data.get('stage', 'unknown'),
                'total_apps': stats.get('total_apps', 0),
                'apps_with_contacts': stats.get('apps_with_contacts', 0),
                'emails_generated': stats.get('emails_generated', 0),
                'emails_sent': stats.get('initial_emails_sent', 0),
                'extraction_percentage': extraction_percentage,
                'generation_percentage': generation_percentage,
                'sending_percentage': sending_percentage,
                'total_emails_sent': total_emails_sent,
                'total_replies': total_replies,
                'reply_rate': reply_rate
            }
            
            # Get apps data for the table
            apps = []
            for i, app in enumerate(campaign_data):
                app_info = {
                    'id': i,
                    'name': app.get('app_name', f'App {i}'),
                    'contact_status': 'has_contacts' if app.get('has_contacts') else 'no_contacts',
                    'contacts_count': len(app.get('extracted_emails', [])),
                    'email_generated': 'generated_emails' in app,
                    'email_status': app.get('email_tracking', {}).get('overall_status', 'pending'),
                    'email_subject': app.get('generated_emails', {}).get('emails', {}).get('subject', ''),
                    'initial_email': app.get('generated_emails', {}).get('emails', {}).get('email1', ''),
                    'followup_emails': [
                        app.get('generated_emails', {}).get('emails', {}).get(f'follow{i}', '')
                        for i in range(1, 4) if f'follow{i}' in app.get('generated_emails', {}).get('emails', {})
                    ],
                    'email_tracking': {
                        'status': app.get('email_tracking', {}).get('overall_status', 'pending'),
                        'initial': app.get('email_tracking', {}).get('email1', {}).get('status', 'pending'),
                        'followup1': app.get('email_tracking', {}).get('follow1', {}).get('status', 'pending'),
                        'followup2': app.get('email_tracking', {}).get('follow2', {}).get('status', 'pending'),
                        'followup3': app.get('email_tracking', {}).get('follow3', {}).get('status', 'pending')
                    }
                }
                apps.append(app_info)
            
            return render_template('new_campaign_details.html', campaign=campaign, apps=apps)
        else:
            return render_template('404.html'), 404
    except Exception as e:
        return render_template('404.html', error=str(e)), 404

@app.route('/campaign/progress/<campaign_id>')
def campaign_progress(campaign_id):
    # Get campaign progress data
    try:
        # Call the campaign_progress API endpoint
        from main import campaign_progress as main_campaign_progress
        
        # Simulate the request with args
        class Args:
            def __init__(self, campaign_id):
                self.args = {'campaign_id': campaign_id}
            def get(self, key):
                return self.args.get(key)
        
        # Store the original request object
        original_request = request
        
        # Replace request with our custom object
        import flask
        flask.request = Args(campaign_id)
        
        # Call the function
        response = main_campaign_progress()
        
        # Restore the original request object
        flask.request = original_request
        
        # Extract the data from the response
        if hasattr(response, 'json') and callable(response.json):
            progress_data = response.json()
        else:
            progress_data = response
        
        # Render the template with the progress data
        return render_template('campaign_progress.html', campaign_id=campaign_id, progress=progress_data)
    except Exception as e:
        return render_template('404.html', error=str(e)), 404

# API Endpoints
@app.route('/campaign/upload', methods=['POST'])
def upload_campaign():
    # Forward the request to main.py's upload_campaign endpoint
    campaign_id = request.form.get('campaign_id')
    
    if 'file' not in request.files:
        return jsonify({"error": "No file part provided in the request."}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected for upload."}), 400
    
    # Import the upload_campaign function from main.py
    from main import upload_campaign as main_upload_campaign
    
    # Call the function directly with the request object
    return main_upload_campaign()

@app.route('/campaign/extract-contacts', methods=['POST'])
def extract_contacts_route():
    # Forward the request to main.py's extract_contacts endpoint
    from main import extract_contacts as main_extract_contacts
    
    # Call the function directly
    return main_extract_contacts()

@app.route('/campaign/<campaign_id>/common-keys', methods=['GET'])
def get_campaign_common_keys(campaign_id):
    # Forward the request to main.py's get_common_keys endpoint
    from main import get_common_keys
    
    # Call the function directly
    return get_common_keys(campaign_id)

@app.route('/campaign/<campaign_id>/apps', methods=['GET'])
def get_campaign_apps(campaign_id):
    # Forward the request to main.py's get_campaign_apps_status endpoint
    from main import get_campaign_apps_status as main_get_campaign_apps
    
    # Call the function directly
    return main_get_campaign_apps(campaign_id)

@app.route('/campaign/generate-preview', methods=['POST'])
def generate_preview():
    # Forward the request to main.py's generate_email_preview endpoint
    from main import generate_email_preview as main_generate_preview
    
    # Call the function directly
    return main_generate_preview()

@app.route('/campaign/generate-emails', methods=['POST'])
def generate_emails():
    # Forward the request to main.py's generate_emails_start endpoint
    from main import generate_emails_start as main_generate_emails
    
    # Call the function directly
    return main_generate_emails()

@app.route('/campaign/send-emails', methods=['POST'])
def send_emails():
    # Forward the request to main.py's send_emails_start endpoint
    from main import send_emails_start as main_send_emails
    
    # Call the function directly
    return main_send_emails()

@app.route('/campaign/progress', methods=['GET'])
def get_campaign_progress():
    # Forward the request to main.py's campaign_progress endpoint
    from main import campaign_progress as main_campaign_progress
    
    # Call the function directly
    return main_campaign_progress()

@app.route('/campaigns', methods=['GET'])
def get_campaigns():
    # Forward the request to main.py's list_campaigns endpoint
    from main import list_campaigns as main_list_campaigns
    
    # Call the function directly
    return main_list_campaigns()

@app.route('/analytics/overall', methods=['GET'])
def get_overall_analytics():
    # Forward the request to main.py's overall_analytics endpoint
    from main import overall_analytics as main_overall_analytics
    
    # Call the function directly
    return main_overall_analytics()

# Error handlers
@app.route('/campaign_details/<campaign_id>')
def detailed_campaign_view(campaign_id):
    """Route for the new detailed campaign view"""
    try:
        from main import calculate_campaign_stats
        campaign_data = load_campaign_data(campaign_id)
        progress_data = load_progress_data(campaign_id)
        
        if campaign_data and progress_data:
            # Calculate stats
            stats = calculate_campaign_stats(campaign_data)
            
            # Calculate percentages for progress bars
            extraction_percentage = 100 if progress_data.get('stage') != 'uploaded' else 0
            generation_percentage = 100 if progress_data.get('stage') in ['emails_generated', 'email_sending', 'followups_active', 'completed'] else 0
            sending_percentage = 100 if progress_data.get('stage') in ['followups_active', 'completed'] else (50 if progress_data.get('stage') == 'email_sending' else 0)
            
            # Get email stats
            total_emails_sent = stats.get('initial_emails_sent', 0) + stats.get('total_followups_sent', 0)
            total_replies = stats.get('total_replies', 0)
            reply_rate = stats.get('reply_rate', 0)
            
            # Combine data
            campaign = {
                'id': campaign_id,
                'status': progress_data.get('stage', 'unknown'),
                'total_apps': stats.get('total_apps', 0),
                'apps_with_contacts': stats.get('apps_with_contacts', 0),
                'emails_generated': stats.get('emails_generated', 0),
                'emails_sent': stats.get('initial_emails_sent', 0),
                'extraction_percentage': extraction_percentage,
                'generation_percentage': generation_percentage,
                'sending_percentage': sending_percentage,
                'total_emails_sent': total_emails_sent,
                'total_replies': total_replies,
                'reply_rate': reply_rate
            }
            
            return render_template('campaign_details_page.html', campaign=campaign, campaign_id=campaign_id)
        else:
            return render_template('404.html'), 404
    except Exception as e:
        return render_template('404.html', error=str(e)), 404

@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404

if __name__ == '__main__':
    import sys
    port = 8080
    if len(sys.argv) > 1 and sys.argv[1] == '--port' and len(sys.argv) > 2:
        port = int(sys.argv[2])
    app.run(debug=True, host='0.0.0.0', port=port)