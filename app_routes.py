from flask import Blueprint, jsonify, request
import main

api_routes = Blueprint('api_routes', __name__)

@api_routes.route('/campaign/<campaign_id>/apps', methods=['GET'])
def get_campaign_apps(campaign_id):
    """
    API endpoint to get campaign details and apps data for the campaign details page
    """
    try:
        # Load campaign data
        campaign_data = main.load_campaign_data(campaign_id)
        if not campaign_data:
            return jsonify({'error': 'Campaign not found'}), 404
        
        # Load progress data
        progress_data = main.load_progress_data(campaign_id)
        
        # Calculate campaign statistics
        stats = main.calculate_campaign_stats(campaign_id)
        
        # Prepare campaign details
        campaign = {
            'id': campaign_id,
            'status': campaign_data.get('status', 'uploaded'),
            'total_apps': len(campaign_data.get('apps', [])),
            'apps_with_contacts': sum(1 for app in campaign_data.get('apps', []) if app.get('contacts')),
            'emails_generated': sum(1 for app in campaign_data.get('apps', []) if app.get('email_generated')),
            'emails_sent': sum(1 for app in campaign_data.get('apps', []) if app.get('email_status') == 'sent' or app.get('email_status') == 'replied'),
            'extraction_percentage': int(progress_data.get('contact_extraction', 0) * 100),
            'generation_percentage': int(progress_data.get('email_generation', 0) * 100),
            'sending_percentage': int(progress_data.get('email_sending', 0) * 100),
            'total_emails_sent': stats.get('emails_sent', 0),
            'replies_received': stats.get('replies', 0),
            'reply_rate': int(stats.get('reply_rate', 0) * 100),
            'initial_emails_sent': stats.get('initial_emails_sent', 0),
            'total_followups_sent': stats.get('followup_emails_sent', 0),
            'pending_emails': stats.get('pending_emails', 0),
            'sent_emails': stats.get('sent_emails', 0),
            'replied_emails': stats.get('replied_emails', 0),
            'failed_emails': stats.get('failed_emails', 0)
        }
        
        # Prepare apps data
        apps = []
        for i, app in enumerate(campaign_data.get('apps', [])):
            # Get email tracking status
            email_tracking = {
                'initial': app.get('email_status', 'none'),
                'followup1': app.get('followup1_status', None),
                'followup2': app.get('followup2_status', None),
                'followup3': app.get('followup3_status', None)
            }
            
            # Get followup emails content
            followup_emails = []
            for i in range(1, 4):
                followup_key = f'followup{i}_email'
                if followup_key in app and app[followup_key]:
                    followup_emails.append(app[followup_key])
            
            app_data = {
                'id': i,  # Using index as ID for simplicity
                'name': app.get('name', ''),
                'contact_status': 'has_contacts' if app.get('contacts') else 'no_contacts',
                'contacts_count': len(app.get('contacts', [])),
                'email_generated': bool(app.get('email_generated')),
                'email_status': app.get('email_status', 'none'),
                'email_subject': app.get('email_subject', ''),
                'initial_email': app.get('email', ''),
                'followup_emails': followup_emails,
                'email_tracking': email_tracking
            }
            apps.append(app_data)
        
        return jsonify({
            'campaign': campaign,
            'apps': apps
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500