import os
import json
import time
import logging
import smtplib
import imaplib
import threading
import shutil
import copy
from enum import Enum
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Any
from collections import Counter
from datetime import datetime, timedelta
from email.mime.text import MIMEText  # <-- IMPORTANT: Added missing import
from json import JSONDecodeError
from dotenv import load_dotenv

from flask import Flask, request, jsonify
from openai import OpenAI
import schedule

# Load environment variables from .env file
load_dotenv()

# ==============================================================================
# 1. SETUP & CONFIGURATION
# ==============================================================================

# --- Basic Setup ---
app = Flask(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Configuration (Load from Environment Variables or hardcode here) ---
DATA_DIR = "campaign_data"
os.makedirs(DATA_DIR, exist_ok=True)
# Initialize OpenAI client with API key
openai_api_key = os.environ.get("OPENAI_API_KEY", "")  # Recommended to use environment variables
# Create OpenAI client instance for v1.3.0+
try:
    # Try to create client without proxies parameter
    import httpx
    http_client = httpx.Client()
    openai_client = OpenAI(api_key=openai_api_key, http_client=http_client)
except Exception as e:
    logger.error(f"Error initializing OpenAI client: {e}")
    # Fallback to default initialization
    openai_client = OpenAI(api_key=openai_api_key)
GMAIL_USER = os.environ.get("GMAIL_USER", "")
GMAIL_PASSWORD = os.environ.get("GMAIL_PASSWORD", "")  # IMPORTANT: Use an App Password from Google
TEST_EMAIL = os.environ.get("TEST_EMAIL", "")
# Email Configuration
EMAIL_SENDER_NAME = os.environ.get("EMAIL_SENDER_NAME", "UndrApp Intel")
EMAIL_REPLY_TO = os.environ.get("EMAIL_REPLY_TO", GMAIL_USER)
# OpenAI Model Configuration
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o")
# Global dictionary to track background tasks (generation, sending)
background_tasks = {}

# --- NEW: Thread-safe locking mechanism for file access ---
# This prevents JSON file corruption from simultaneous writes by different threads.
_campaign_locks = {}
_dict_lock = threading.Lock()

def get_campaign_lock(campaign_id: str) -> threading.Lock:
    """Gets a unique lock for a given campaign ID to ensure thread safety."""
    with _dict_lock:
        if campaign_id not in _campaign_locks:
            _campaign_locks[campaign_id] = threading.Lock()
        return _campaign_locks[campaign_id]

# ==============================================================================
# 2. ENUMS & DATACLASSES
# ==============================================================================

class CampaignStatus(Enum):
    UPLOADED = "uploaded"; EXTRACTION_DONE = "extraction_done"; GENERATING_EMAILS = "generating_emails"; EMAILS_GENERATED = "emails_generated"; EMAIL_SENDING = "email_sending"; FOLLOWUPS_ACTIVE = "followups_active"; COMPLETED = "completed"; FAILED = "failed"

class EmailStatus(Enum):
    PENDING = "pending"; SENT = "sent"; REPLIED = "replied"; FAILED = "failed"; SKIPPED_REPLY = "skipped_due_to_reply"

@dataclass
class EmailConfig:
    followup_delay_hours: Optional[int]
    followup_delay_minutes: Optional[int]
    max_followups: int
    test_mode: bool
    sender_name: str
    sender_company: str
    test_email: Optional[str] = None

# ==============================================================================
# 3. HELPER & CORE LOGIC FUNCTIONS
# ==============================================================================

def get_campaign_path(campaign_id: str, filename: str) -> str: return os.path.join(DATA_DIR, campaign_id, filename)
def ensure_campaign_folder(campaign_id: str) -> None: os.makedirs(os.path.join(DATA_DIR, campaign_id), exist_ok=True)
def save_campaign_data(campaign_id: str, data: List[Dict], filename: str = "campaign_data.json") -> None:
    with open(get_campaign_path(campaign_id, filename), 'w') as f: json.dump(data, f, indent=2)
def load_campaign_data(campaign_id: str, filename: str = "campaign_data.json") -> Optional[List[Dict]]:
    path = get_campaign_path(campaign_id, filename)
    if not os.path.exists(path): return None
    with open(path, 'r') as f: return json.load(f)
def load_progress_data(campaign_id: str) -> Optional[Dict]:
    path = get_campaign_path(campaign_id, "progress.json")
    if not os.path.exists(path): return None
    with open(path, 'r') as f: return json.load(f)
def update_progress(campaign_id: str, stage: CampaignStatus, **kwargs) -> None:
    try:
        progress_path = get_campaign_path(campaign_id, "progress.json")
        progress = load_progress_data(campaign_id) or {}
        progress["stage"] = stage.value; progress["last_updated"] = datetime.now().isoformat()
        progress.update(kwargs)
        with open(progress_path, 'w') as f: json.dump(progress, f, indent=2)
    except Exception as e: logger.error(f"Failed to update progress for {campaign_id}: {e}")
def send_email_smtp(to_email: str, subject: str, body: str, sender_name: str, test_mode: bool = False, test_email: Optional[str] = None) -> bool:
    if not GMAIL_USER or not GMAIL_PASSWORD: logger.error("Gmail credentials not configured."); return False
    actual_to_email = test_email or TEST_EMAIL if test_mode else to_email
    try:
        msg = MIMEText(body, 'plain', 'utf-8'); msg['From'] = f"{sender_name} <{GMAIL_USER}>"; msg['To'] = actual_to_email
        msg['Subject'] = f"[TEST] {subject}" if test_mode else subject
        msg['Reply-To'] = EMAIL_REPLY_TO
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
            server.login(GMAIL_USER, GMAIL_PASSWORD); server.send_message(msg)
        logger.info(f"Email sent to {actual_to_email} (Original: {to_email}, Test: {test_mode})")
        return True
    except Exception as e: logger.error(f"Failed to send email to {to_email}: {e}"); return False
def check_for_replies(contact_emails: List[str], since_date: datetime) -> bool:
    if not GMAIL_USER or not GMAIL_PASSWORD: logger.warning("IMAP credentials not set."); return False
    try:
        with imaplib.IMAP4_SSL('imap.gmail.com') as mail:
            mail.login(GMAIL_USER, GMAIL_PASSWORD); mail.select('inbox', readonly=True)
            since_str = since_date.strftime("%d-%b-%Y")
            for email_addr in contact_emails:
                search_criteria = f'FROM "{email_addr}" SINCE "{since_str}"'
                result, messages = mail.search('US-ASCII', search_criteria)
                if result == 'OK' and messages[0]: logger.info(f"REPLY DETECTED from {email_addr}."); return True
        return False
    except Exception as e: logger.error(f"Failed to check for replies: {e}"); return False

@app.route('/campaign/<campaign_id>/common-keys', methods=['GET'])
def get_common_keys(campaign_id):
    """
    Analyzes the campaign data to find the most common top-level keys.
    Useful for helping the user select which keys to use for AI generation.
    """
    try:
        # We use the raw_data because it's the original, unaltered data structure.
        data = load_campaign_data(campaign_id, "raw_data.json")
        if data is None:
            return jsonify({"error": "Campaign data not found. Please upload first."}), 404

        if not isinstance(data, list) or not data:
            return jsonify({"error": "Campaign data is empty or not in the expected format (list of objects)."}), 400

        # Use Counter to efficiently count all keys from all app objects
        key_counter = Counter()
        for app_object in data:
            if isinstance(app_object, dict):
                key_counter.update(app_object.keys())

        # The most_common() method returns a list of (key, count) tuples, sorted by count
        sorted_keys = key_counter.most_common()

        return jsonify({
            "campaign_id": campaign_id,
            "total_apps_analyzed": len(data),
            "unique_key_count": len(sorted_keys),
            "common_keys": [
                {"key": key, "occurrence_count": count, "percentage": round((count / len(data)) * 100, 2)}
                for key, count in sorted_keys
            ]
        })

    except Exception as e:
        logger.error(f"Failed to get common keys for {campaign_id}: {e}")
        return jsonify({"error": "An internal server error occurred."}), 500


def update_app_email_status(campaign_id: str, app_index: int, email_type: str, status: EmailStatus, error_msg: Optional[str] = None):
    # FIX: Use a lock to make the read-modify-write operation atomic and prevent file corruption.
    lock = get_campaign_lock(campaign_id)
    with lock:
        try:
            data = load_campaign_data(campaign_id)
            if not data or app_index >= len(data): return
            app = data[app_index]
            if 'email_tracking' not in app: app['email_tracking'] = {}
            app['email_tracking'][email_type] = {'status': status.value, 'timestamp': datetime.now().isoformat(), 'error': error_msg}
            if status in [EmailStatus.REPLIED, EmailStatus.SKIPPED_REPLY]:
                app['email_tracking']['overall_status'] = EmailStatus.REPLIED.value
            elif app['email_tracking'].get('overall_status') != EmailStatus.REPLIED.value:
                num_followups = app.get("generated_emails", {}).get("num_followups", 0)
                is_last_followup_sent = email_type == f"follow{num_followups}" and status == EmailStatus.SENT
                if is_last_followup_sent: app['email_tracking']['overall_status'] = 'completed_no_reply'
                else: app['email_tracking']['overall_status'] = 'in_progress'
            save_campaign_data(campaign_id, data)
        except Exception as e: logger.error(f"Failed to update app email status for {campaign_id}, app {app_index}: {e}")

def calculate_campaign_stats(data: List[Dict]) -> Dict:
    stats = Counter(); stats["total_apps"] = len(data)
    for app in data:
        if not app.get("has_contacts"): continue
        stats["apps_with_contacts"] += 1
        tracking = app.get("email_tracking", {})
        overall_status = tracking.get("overall_status")
        if overall_status == EmailStatus.REPLIED.value: stats["total_replies"] += 1
        if overall_status in [EmailStatus.REPLIED.value, "completed_no_reply"]: stats["apps_completed"] += 1
        if tracking.get("email1", {}).get("status") == EmailStatus.SENT.value: stats["initial_emails_sent"] += 1
        for key, value in tracking.items():
            if "follow" in key and isinstance(value, dict) and value.get("status") == EmailStatus.SENT.value:
                stats["total_followups_sent"] += 1
    stats["reply_rate"] = round((stats["total_replies"] / stats["initial_emails_sent"]) * 100, 2) if stats["initial_emails_sent"] > 0 else 0
    stats["completion_rate"] = round((stats["apps_completed"] / stats["apps_with_contacts"]) * 100, 2) if stats["apps_with_contacts"] > 0 else 0
    return dict(stats)

# ==============================================================================
# 4. BACKGROUND TASK WORKERS
# ==============================================================================

def generate_emails_logic(app_data_full: Dict, selected_keys: List[str], tone: str, num_followups: int) -> Dict:
    system_prompt = f"""
    You are an expert SaaS sales representative for UndrAds, an AI-powered ad monetization platform.
    Your task is to create a personalized, engaging email sequence.
    - Analyze the provided app data to personalize emails.
    - Highlight how UndrAds automates ad operations and optimizes revenue.
    - Use a {tone} tone.
    - Create {num_followups + 1} emails total: 1 initial outreach + {num_followups} follow-ups.
    - Respond ONLY with valid JSON with keys: "subject", "email1", and "follow1", "follow2", etc., up to "follow{num_followups}".
    """
    app_data_for_gpt = {key: value for key, value in app_data_full.items() if key in selected_keys}
    essential_fields = {"app_name": app_data_full.get("app_name"), "extracted_emails": app_data_full.get("extracted_emails")}
    app_data_for_gpt.update(essential_fields)
    response = openai_client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": f"App Data: {json.dumps(app_data_for_gpt, indent=2)}"}],
        temperature=0.7, response_format={"type": "json_object"})
    parsed_emails = json.loads(response.choices[0].message.content)
    required = ["subject", "email1"] + [f"follow{i + 1}" for i in range(num_followups)]
    if not all(field in parsed_emails and parsed_emails[field] for field in required):
        raise ValueError("Missing or empty required fields in OpenAI response.")
    return parsed_emails


def generate_preview(prompt: str, contact_data: Dict[str, str], common_keys: List[str]) -> str:
    """Generate a preview email for a single contact."""
    try:
        # Create a personalized system message with contact data
        system_message = f"""You are an AI assistant that writes personalized emails. 
        Use the following information about the recipient to personalize the email:
        """
        
        # Add contact information to system message
        for key, value in contact_data.items():
            if key in common_keys and value.strip():  # Only include common keys with non-empty values
                system_message += f"\n{key}: {value}"
        
        # Add the user's prompt
        user_message = prompt
        
        # Generate email using OpenAI
        response = openai_client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message}
            ],
            temperature=0.7,
        )
        
        # Extract the generated email
        email_content = response.choices[0].message.content
        return email_content
        
    except Exception as e:
        logger.error(f"Error in generate_preview: {e}")
        return f"Error generating preview: {str(e)}"


def generate_emails_background(campaign_id: str, limit: Optional[int], tone: str, num_followups: int,
                               selected_keys: List[str], selected_emails: Dict[str, List[str]] = None):
    lock = get_campaign_lock(campaign_id)
    try:
        # Load data once ONLY to determine the list of work to do.
        # This data will NOT be used for writing.
        initial_data = load_campaign_data(campaign_id)
        if not initial_data:
            raise FileNotFoundError("Campaign data not found.")

        apps_to_process = [(i, app) for i, app in enumerate(initial_data) if
                           app.get("has_contacts") and "generated_emails" not in app]
        if limit:
            apps_to_process = apps_to_process[:limit]

        total_to_generate = len(apps_to_process)
        update_progress(campaign_id, CampaignStatus.GENERATING_EMAILS, status="in_progress",
                        total_pending=total_to_generate, total_done=0, total_failed=0)

        generated_count, failed_count = 0, 0
        selected_emails = selected_emails or {}

        # --- CORRECTED LOGIC ---
        # Loop through the app list, but perform a full, atomic read-modify-write
        # cycle inside the lock for EACH app to prevent data loss and corruption.
        for app_index, app_data_for_generation in apps_to_process:
            try:
                # Check if we have selected emails for this app
                app_selected_emails = selected_emails.get(str(app_index), [])
                
                # If selected emails are provided, temporarily replace the extracted_emails in the app data
                if app_selected_emails:
                    app_copy = copy.deepcopy(app_data_for_generation)
                    app_copy['extracted_emails'] = app_selected_emails
                    # 1. Perform the slow API call OUTSIDE the lock to not block other threads.
                    parsed_emails = generate_emails_logic(app_copy, selected_keys, tone, num_followups)
                else:
                    # 1. Perform the slow API call OUTSIDE the lock to not block other threads.
                    parsed_emails = generate_emails_logic(app_data_for_generation, selected_keys, tone, num_followups)

                # 2. Acquire the lock to perform a safe, atomic update.
                with lock:
                    # 3. Re-load the LATEST data from disk right before modifying it.
                    current_data = load_campaign_data(campaign_id)
                    if not current_data or app_index >= len(current_data):
                        raise IndexError(f"Campaign data changed during processing; skipping app index {app_index}.")

                    # 4. Modify the specific app's data in the LATEST version.
                    current_data[app_index]["generated_emails"] = {
                        "status": "success",
                        "generated_at": datetime.now().isoformat(),
                        "tone": tone,
                        "num_followups": num_followups,
                        "selected_keys": selected_keys,
                        "emails": parsed_emails
                    }
                    
                    # Store the selected emails if provided
                    if app_selected_emails:
                        current_data[app_index]["generated_emails"]["selected_emails"] = app_selected_emails
                    
                    # 5. Save the updated data back to disk immediately.
                    save_campaign_data(campaign_id, current_data)

                generated_count += 1
                logger.info(f"Successfully generated and saved emails for app index: {app_index}")

            except Exception as e:
                logger.error(f"Email generation error for app index {app_index}: {e}")
                # Also lock the update for a failed status to be safe.
                with lock:
                    # Re-read the file even for failures to avoid overwriting good data.
                    current_data = load_campaign_data(campaign_id)
                    if current_data and app_index < len(current_data):
                        current_data[app_index]["generated_emails"] = {"status": "failed", "error": str(e)}
                        save_campaign_data(campaign_id, current_data)
                failed_count += 1

            # Update non-critical progress file outside the main data lock.
            update_progress(campaign_id, CampaignStatus.GENERATING_EMAILS, status="in_progress",
                            total_done=generated_count, total_failed=failed_count,
                            total_pending=total_to_generate - (generated_count + failed_count))

        update_progress(campaign_id, CampaignStatus.EMAILS_GENERATED, status="completed", total_done=generated_count,
                        total_failed=failed_count)

    except Exception as e:
        logger.error(f"Major background email generation failure for {campaign_id}: {e}")
        update_progress(campaign_id, CampaignStatus.FAILED, error=str(e))
    finally:
        if campaign_id in background_tasks:
            del background_tasks[campaign_id]

def send_emails_background(campaign_id: str, config: EmailConfig, limit: Optional[int] = None):
    try:
        update_progress(campaign_id, CampaignStatus.EMAIL_SENDING, status="in_progress", config=asdict(config))
        data = load_campaign_data(campaign_id)
        progress = load_progress_data(campaign_id)
        if not data or not progress: raise FileNotFoundError("Campaign data or progress not found.")
        campaign_start_date = datetime.fromisoformat(progress.get('last_updated', datetime.now().isoformat()))
        apps_to_process = [(i, app) for i, app in enumerate(data) if app.get("has_contacts") and app.get("generated_emails", {}).get("status") == "success" and app.get("email_tracking", {}).get("overall_status") == EmailStatus.PENDING.value]
        if limit: apps_to_process = apps_to_process[:limit]
        sent_count, failed_count, skipped_count = 0, 0, 0
        for app_index, app in apps_to_process:
            try:
                # Get the emails to send to - either selected emails or all extracted emails
                emails_to_send = app["generated_emails"].get("selected_emails", app["extracted_emails"])
                
                if not emails_to_send:
                    logger.info(f"No emails to send for app {app_index}. Skipping.")
                    update_app_email_status(campaign_id, app_index, "email1", EmailStatus.SKIPPED_REPLY)
                    skipped_count += 1; continue
                    
                if check_for_replies(emails_to_send, campaign_start_date):
                    logger.info(f"Reply detected for app {app_index} before initial send. Skipping.")
                    update_app_email_status(campaign_id, app_index, "email1", EmailStatus.SKIPPED_REPLY)
                    skipped_count += 1; continue
                    
                subject, body = app["generated_emails"]["emails"]["subject"], app["generated_emails"]["emails"]["email1"]
                all_sent_successfully = all(send_email_smtp(email, subject, body, config.sender_name, config.test_mode, config.test_email) for email in emails_to_send)
                if all_sent_successfully:
                    update_app_email_status(campaign_id, app_index, "email1", EmailStatus.SENT); sent_count += 1
                    if config.max_followups > 0: schedule_followup(campaign_id, app_index, 1, config)
                else: update_app_email_status(campaign_id, app_index, "email1", EmailStatus.FAILED, "SMTP Error"); failed_count += 1
            except Exception as e:
                logger.error(f"Error processing initial email for app {app_index}: {e}")
                update_app_email_status(campaign_id, app_index, "email1", EmailStatus.FAILED, str(e)); failed_count += 1
        new_status = CampaignStatus.FOLLOWUPS_ACTIVE if config.max_followups > 0 else CampaignStatus.COMPLETED
        update_progress(campaign_id, new_status, status="completed", initial_emails_sent=sent_count, initial_emails_failed=failed_count, initial_emails_skipped=skipped_count)
    except Exception as e: logger.error(f"Background email sending failed for {campaign_id}: {e}"); update_progress(campaign_id, CampaignStatus.FAILED, error=str(e))
    finally:
        if campaign_id in background_tasks: del background_tasks[campaign_id]

def schedule_followup(campaign_id: str, app_index: int, followup_num: int, config: EmailConfig):
    def send_followup_task():
        try:
            # Note: update_app_email_status is already thread-safe due to the lock inside it.
            data = load_campaign_data(campaign_id)
            if not data or app_index >= len(data): return
            app = data[app_index]
            if app.get("email_tracking", {}).get("overall_status") == EmailStatus.REPLIED.value:
                logger.info(f"Follow-up for app {app_index} stopped due to reply."); return
                
            # Get the emails to send to - either selected emails or all extracted emails
            emails_to_send = app.get("generated_emails", {}).get("selected_emails", app.get("extracted_emails", []))
            if not emails_to_send:
                logger.info(f"No emails to send for app {app_index}. Skipping follow-up {followup_num}.")
                update_app_email_status(campaign_id, app_index, f"follow{followup_num}", EmailStatus.SKIPPED_REPLY); return
                
            last_sent_key = f"follow{followup_num - 1}" if followup_num > 1 else "email1"
            last_sent_info = app.get("email_tracking", {}).get(last_sent_key, {})
            if last_sent_info.get("status") == EmailStatus.SENT.value:
                since_date = datetime.fromisoformat(last_sent_info["timestamp"])
                if check_for_replies(emails_to_send, since_date):
                    logger.info(f"Reply received for app {app_index}, skipping follow-up {followup_num}.")
                    update_app_email_status(campaign_id, app_index, f"follow{followup_num}", EmailStatus.SKIPPED_REPLY); return
            emails, followup_key = app.get("generated_emails", {}).get("emails", {}), f"follow{followup_num}"
            if followup_key not in emails: return
            subject, body = f"Re: {emails.get('subject', 'Following up')}", emails[followup_key]
            all_sent_successfully = all(send_email_smtp(email, subject, body, config.sender_name, config.test_mode, config.test_email) for email in emails_to_send)
            if all_sent_successfully:
                update_app_email_status(campaign_id, app_index, followup_key, EmailStatus.SENT)
                if followup_num < config.max_followups: schedule_followup(campaign_id, app_index, followup_num + 1, config)
            else: update_app_email_status(campaign_id, app_index, followup_key, EmailStatus.FAILED, "SMTP Error")
        except Exception as e:
            logger.error(f"Error in follow-up task {followup_num} for app {app_index}: {e}")
            update_app_email_status(campaign_id, app_index, f"follow{followup_num}", EmailStatus.FAILED, str(e))

    total_delay_seconds = (config.followup_delay_hours or 0) * 3600 + (config.followup_delay_minutes or 0) * 60
    if total_delay_seconds <= 0: logger.warning(f"Follow-up for app {app_index} has zero delay. Skipping."); return
    timer = threading.Timer(total_delay_seconds, send_followup_task)
    timer.daemon = True; timer.start()
    logger.info(f"Scheduled follow-up #{followup_num} for app {app_index} in {total_delay_seconds} seconds.")

def validate_json_structure(data: Any) -> bool:
    return isinstance(data, list) and all(isinstance(item, dict) for item in data)
# ==============================================================================
# 5. FLASK API ENDPOINTS
# ==============================================================================
@app.route('/campaign/upload', methods=['POST'])
def upload_campaign():
    campaign_id = request.form.get('campaign_id') or f"Campaign_{int(time.time())}"
    ensure_campaign_folder(campaign_id)

    if 'file' not in request.files:
        return jsonify({"error": "No file part provided in the request."}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected for upload."}), 400

    try:
        # Step 1: Try to parse the JSON data from the file.
        # This will fail if the file content is not valid JSON.
        data = json.load(file)

        # Step 2: If parsing succeeds, validate the structure of the parsed data.
        if not validate_json_structure(data):
            logger.error(f"Structural validation failed for campaign '{campaign_id}'. Data was not a list of objects.")
            return jsonify({
                "error": "Invalid data structure.",
                "detail": "The uploaded file must be a valid JSON array (starts with '[') where each element is a JSON object (starts with '{')."
            }), 400

        # If both parsing and validation succeed, proceed.
        save_campaign_data(campaign_id, data, "raw_data.json")
        update_progress(campaign_id, CampaignStatus.UPLOADED, total_apps=len(data))
        return jsonify(
            {"message": "Campaign uploaded successfully", "campaign_id": campaign_id, "total_apps": len(data)})

    except JSONDecodeError as e:
        # This block catches errors from `json.load()` if the file is not a valid JSON document.
        logger.error(f"JSON parsing failed for campaign upload: {e}")
        return jsonify({
            "error": "Invalid JSON format.",
            "detail": f"The uploaded file could not be parsed as JSON. Please check for syntax errors. Parser error: {e}"
        }), 400
    except Exception as e:
        # This is a catch-all for other unexpected errors (e.g., file not readable).
        logger.error(f"An unexpected error occurred during upload for campaign '{campaign_id}': {e}")
        return jsonify({"error": "An unexpected server error occurred during upload."}), 500

@app.route('/campaign/extract-contacts', methods=['POST'])
def extract_contacts():
    campaign_id = request.json.get('campaign_id')
    if not campaign_id: return jsonify({"error": "campaign_id is required"}), 400
    # FIX: Lock the read-modify-write operation
    lock = get_campaign_lock(campaign_id)
    with lock:
        try:
            data = load_campaign_data(campaign_id, "raw_data.json")
            if data is None: return jsonify({"error": "Campaign not found"}), 404
            with_contacts_count = 0
            for app in data:
                emails = set(e.strip().lower() for e in str(app.get("publisher_contacts", "")).split(',') if "@" in e)
                dev_email = app.get("developer_info", {}).get("email")
                if dev_email and '@' in dev_email: emails.add(dev_email.lower())
                app["extracted_emails"] = sorted(list(emails))
                app["has_contacts"] = bool(app["extracted_emails"])
                if app["has_contacts"]: app["email_tracking"] = {"overall_status": EmailStatus.PENDING.value}; with_contacts_count += 1
                else: app["email_tracking"] = {"overall_status": "no_contacts"}
            save_campaign_data(campaign_id, data)
            update_progress(campaign_id, CampaignStatus.EXTRACTION_DONE, apps_with_contacts=with_contacts_count)
            return jsonify({"message": "Contact extraction complete", "apps_with_contacts": with_contacts_count})
        except Exception as e: logger.error(f"Extraction error: {e}"); return jsonify({"error": f"Extraction failed: {e}"}), 500

@app.route('/campaign/generate-preview', methods=['POST'])
def generate_email_preview():
    params = request.json
    campaign_id, app_indices = params.get('campaign_id'), params.get('app_indices')
    if not all([campaign_id, app_indices]): return jsonify({"error": "campaign_id and app_indices required"}), 400
    try:
        data = load_campaign_data(campaign_id);
        if not data: return jsonify({"error": "Campaign not found"}), 404
        previews = []
        for index in app_indices:
            if 0 <= index < len(data):
                app = data[index]
                if not app.get('has_contacts'): previews.append({"app_index": index, "status": "skipped", "reason": "No contacts"}); continue
                try:
                    # Get selected emails for this app if provided, otherwise use all extracted emails
                    selected_emails_dict = params.get('selected_emails', {})
                    selected_emails = selected_emails_dict[str(index)] if str(index) in selected_emails_dict else []
                    
                    # If selected emails are provided, temporarily replace the extracted_emails in the app data
                    original_emails = app.get('extracted_emails', [])
                    if selected_emails:
                        app_copy = copy.deepcopy(app)
                        app_copy['extracted_emails'] = selected_emails
                        content = generate_emails_logic(app_copy, params.get('selected_keys', []), params.get('tone', 'conversational'), params.get('num_followups', 3))
                    else:
                        content = generate_emails_logic(app, params.get('selected_keys', []), params.get('tone', 'conversational'), params.get('num_followups', 3))
                    
                    previews.append({"app_index": index, "status": "success", "content": content})
                except Exception as e: logger.error(f"Preview failed for app {index}: {e}"); previews.append({"app_index": index, "status": "failed", "error": str(e)})
            else: previews.append({"app_index": index, "status": "error", "reason": "Index out of bounds"})
        return jsonify({"previews": previews})
    except Exception as e: logger.error(f"Preview endpoint error: {e}"); return jsonify({"error": f"An error occurred: {e}"}), 500

@app.route('/campaign/generate-emails', methods=['POST'])
def generate_emails_start():
    params = request.json
    campaign_id = params.get('campaign_id')
    if not campaign_id: return jsonify({"error": "campaign_id is required"}), 400
    if campaign_id in background_tasks: return jsonify({"message": "A task is already running for this campaign"}), 409
    thread = threading.Thread(target=generate_emails_background, args=(campaign_id, params.get('limit'), params.get('tone', 'conversational'), params.get('num_followups', 3), params.get('selected_keys', []), params.get('selected_emails', {})))
    thread.daemon = True; thread.start()
    background_tasks[campaign_id] = {"thread": thread, "type": "generation", "started_at": time.time()}
    return jsonify({"message": "Email generation started in background"}), 202

@app.route('/campaign/send-emails', methods=['POST'])
def send_emails_start():
    params = request.json
    campaign_id = params.get('campaign_id')
    if not campaign_id: return jsonify({"error": "campaign_id is required"}), 400
    
    # Check if there's a generation task running
    if campaign_id in background_tasks:
        task_info = background_tasks[campaign_id]
        if task_info["type"] == "generation":
            return jsonify({
                "error": "Email generation is still in progress. Please wait until generation is complete before sending emails.",
                "success": False
            }), 409
        elif task_info["type"] == "sending":
            return jsonify({
                "error": "Emails are already being sent for this campaign.",
                "success": False
            }), 409
    
    # Check if all apps in the campaign have their emails generated
    data = load_campaign_data(campaign_id)
    if not data: return jsonify({"error": "Campaign data not found", "success": False}), 404
    
    # Get all apps that have contacts but don't have successfully generated emails
    apps_with_contacts = [app for app in data if app.get("has_contacts", False)]
    if not apps_with_contacts: return jsonify({"error": "No apps with contacts found in this campaign", "success": False}), 400
    
    apps_without_emails = [app for app in apps_with_contacts if not app.get("generated_emails", {}).get("status") == "success"]
    if apps_without_emails:
        return jsonify({
            "error": "Cannot send emails because not all apps have their emails generated",
            "apps_with_contacts": len(apps_with_contacts),
            "apps_without_emails": len(apps_without_emails),
            "success": False
        }), 400
    
    try:
        # Get test_email if test_mode is True
        test_email = params.get('test_email') if params.get('test_mode', False) else None
        if params.get('test_mode', False) and not test_email:
            return jsonify({"error": "Test email is required when test mode is enabled", "success": False}), 400
            
        config = EmailConfig(
            followup_delay_hours=params.get('followup_delay_hours', 0),
            followup_delay_minutes=params.get('followup_delay_minutes', 5),
            max_followups=int(params.get('max_followups', 3)),
            test_mode=bool(params.get('test_mode', False)),
            test_email=test_email,
            sender_name=params.get('sender_name', 'The UndrAds Team'),
            sender_company=params.get('sender_company', 'UndrAds'))
        
        # Ensure we have some delay for followups
        total_delay_seconds = (config.followup_delay_hours or 0) * 3600 + (config.followup_delay_minutes or 0) * 60
        if total_delay_seconds <= 0:
            return jsonify({"error": "You must provide a positive delay for follow-ups", "success": False}), 400
            
    except (KeyError, ValueError) as e: 
        return jsonify({"error": f"Invalid or missing parameter: {e}", "success": False}), 400
        
    # Update campaign status to indicate emails are being sent
    update_progress(campaign_id, CampaignStatus.EMAIL_SENDING)
    
    # Start the email sending in a background thread
    thread = threading.Thread(target=send_emails_background, args=(campaign_id, config, params.get('limit')))
    thread.daemon = True; thread.start()
    background_tasks[campaign_id] = {"thread": thread, "type": "sending", "started_at": time.time()}
    
    return jsonify({"message": "Email sending campaign started in background", "success": True}), 202

@app.route('/campaign/progress', methods=['GET'])
def campaign_progress():
    campaign_id = request.args.get('campaign_id')
    if not campaign_id: return jsonify({"error": "campaign_id is required"}), 400
    progress = load_progress_data(campaign_id)
    if not progress: return jsonify({"error": "Campaign progress not found"}), 404
    if campaign_id in background_tasks:
        progress["background_task_active"] = True
        progress["background_task_details"] = {"type": background_tasks[campaign_id]["type"], "started_at": background_tasks[campaign_id]["started_at"]}
    try:
        data = load_campaign_data(campaign_id)
        if data: progress["live_stats"] = calculate_campaign_stats(data)
    except Exception as e: logger.warning(f"Could not calculate live stats for {campaign_id}: {e}")
    return jsonify(progress)

@app.route('/campaign/<campaign_id>/apps', methods=['GET'])
def get_campaign_apps_status(campaign_id):
    data = load_campaign_data(campaign_id)
    if data is None: return jsonify({"error": "Campaign not found"}), 404
    app_statuses = [{"app_index": i, "app_name": app.get("app_name", "N/A"), "has_contacts": app.get("has_contacts", False), "extracted_emails": app.get("extracted_emails", []), "generation_status": app.get("generated_emails", {}).get("status", "pending"), "tracking": app.get("email_tracking", {})} for i, app in enumerate(data)]
    return jsonify({"campaign_id": campaign_id, "apps": app_statuses})

@app.route('/campaigns', methods=['GET'])
def list_campaigns():
    try:
        campaigns = []
        for campaign_id in os.listdir(DATA_DIR):
            if os.path.isdir(os.path.join(DATA_DIR, campaign_id)):
                campaign_info = {"campaign_id": campaign_id}
                progress = load_progress_data(campaign_id)
                if progress: campaign_info.update(progress)
                try:
                    data = load_campaign_data(campaign_id)
                    if data: campaign_info["stats_summary"] = calculate_campaign_stats(data)
                except: campaign_info["stats_summary"] = "Error loading stats"
                campaigns.append(campaign_info)
        campaigns.sort(key=lambda c: c.get('last_updated', '0'), reverse=True)
        return jsonify({"campaigns": campaigns})
    except Exception as e: logger.error(f"List campaigns error: {e}"); return jsonify({"error": f"Failed to list campaigns: {e}"}), 500

@app.route('/api/dashboard-data', methods=['GET'])
def get_dashboard_data():
    """API endpoint to provide all data needed for the dashboard homepage."""
    try:
        # Get all campaigns
        campaigns = []
        total_stats = {
            'total_campaigns': 0,
            'total_emails_sent': 0,
            'reply_rate': 0,
            'total_apps': 0,
            'campaign_growth': 15,  # Mock data for now
            'email_growth': 8,      # Mock data for now
            'reply_rate_growth': -2, # Mock data for now
            'app_growth': 10        # Mock data for now
        }
        
        # Get list of campaign directories
        for campaign_id in os.listdir(DATA_DIR):
            campaign_dir = os.path.join(DATA_DIR, campaign_id)
            if os.path.isdir(campaign_dir):
                # Load campaign progress
                progress = load_progress_data(campaign_id)
                if progress:
                    campaign_info = {'id': campaign_id}
                    campaign_info.update(progress)
                    
                    # Load campaign data for stats
                    try:
                        campaign_data = load_campaign_data(campaign_id)
                        if campaign_data:
                            campaign_info['stats_summary'] = calculate_campaign_stats(campaign_data)
                            
                            # Update total stats
                            total_stats['total_campaigns'] += 1
                            total_stats['total_emails_sent'] += campaign_info['stats_summary'].get('initial_emails_sent', 0)
                            total_stats['total_apps'] += campaign_info['stats_summary'].get('total_apps', 0)
                            
                            # Add to reply count for calculating overall rate
                            if 'total_replies' in campaign_info['stats_summary'] and 'initial_emails_sent' in campaign_info['stats_summary']:
                                if campaign_info['stats_summary']['initial_emails_sent'] > 0:
                                    reply_rate = campaign_info['stats_summary']['total_replies'] / campaign_info['stats_summary']['initial_emails_sent'] * 100
                                    campaign_info['reply_rate'] = round(reply_rate, 2)
                                    
                            # Add additional fields needed for dashboard
                            campaign_info['total_apps'] = campaign_info['stats_summary'].get('total_apps', 0)
                            campaign_info['apps_with_contacts'] = campaign_info['stats_summary'].get('apps_with_contacts', 0)
                            campaign_info['initial_emails_sent'] = campaign_info['stats_summary'].get('initial_emails_sent', 0)
                    except Exception as e:
                        logger.error(f"Error calculating stats for {campaign_id}: {e}")
                        campaign_info['stats_summary'] = {'error': str(e)}
                    
                    campaigns.append(campaign_info)
        
        # Calculate overall reply rate
        if total_stats['total_emails_sent'] > 0:
            total_stats['reply_rate'] = round(sum(c['stats_summary'].get('total_replies', 0) for c in campaigns if 'stats_summary' in c) / total_stats['total_emails_sent'] * 100, 2)
        
        # Sort campaigns by last updated
        campaigns.sort(key=lambda c: c.get('last_updated', '0'), reverse=True)
        
        return jsonify({
            'campaigns': campaigns,
            'stats': total_stats
        })
    except Exception as e:
        logger.error(f"Dashboard data error: {e}")
        return jsonify({
            'error': str(e),
            'campaigns': [],
            'stats': {
                'total_campaigns': 0,
                'total_emails_sent': 0,
                'reply_rate': 0,
                'total_apps': 0,
                'campaign_growth': 0,
                'email_growth': 0,
                'reply_rate_growth': 0,
                'app_growth': 0
            }
        }), 500

@app.route('/analytics/overall', methods=['GET'])
def overall_analytics():
    total_stats, campaign_details = Counter(), []
    email_sequence_stats = Counter()
    email_sequence_replies = Counter()
    email_status_stats = Counter()
    
    # Count of campaigns
    total_stats["total_campaigns"] = 0
    
    # Historical data for growth calculations
    # In a real implementation, this would come from a database with timestamped records
    # For now, we'll use mock data for demonstration
    historical_data = {
        "previous_period": {
            "total_campaigns": 8,  # Previous period had 8 campaigns
            "total_emails_sent": 320,  # Previous period sent 320 emails
            "total_replies": 35,  # Previous period had 35 replies
            "total_apps": 180  # Previous period had 180 apps
        }
    }
    
    for campaign_id in os.listdir(DATA_DIR):
        if not os.path.isdir(os.path.join(DATA_DIR, campaign_id)): continue
        try:
            data = load_campaign_data(campaign_id)
            if not data: continue
            
            # Increment campaign count
            total_stats["total_campaigns"] += 1
            
            # Calculate campaign stats and handle errors
            try:
                stats = calculate_campaign_stats(data)
                
                # Add creation date for sorting and filtering
                progress = load_progress_data(campaign_id)
                creation_date = progress.get('last_updated') if progress else datetime.now().isoformat()
                
                # Add campaign status
                status = progress.get('stage') if progress else 'unknown'
                
                # Add to campaign details with additional fields
                campaign_details.append({
                    "campaign_id": campaign_id, 
                    "creation_date": creation_date,
                    "status": status,
                    "emails_sent": stats.get("initial_emails_sent", 0),
                    "opened": stats.get("initial_emails_sent", 0) * 0.7,  # Mock data for opened emails (70% open rate)
                    "total_replies": stats.get("total_replies", 0),
                    "reply_rate": stats.get("reply_rate", 0),
                    **stats
                })
                
                # Aggregate stats
                for key, value in stats.items():
                    if key not in ["reply_rate", "completion_rate"]: total_stats[key] += value
            except Exception as stats_error:
                logger.error(f"Error calculating stats for campaign {campaign_id}: {stats_error}")
                # Add campaign with error info but don't include in aggregated stats
                campaign_details.append({"campaign_id": campaign_id, "error": str(stats_error)})
                continue
            
            # Calculate email sequence stats
            for app in data:
                if not isinstance(app, dict):
                    logger.warning(f"Skipping non-dictionary app data in campaign {campaign_id}")
                    continue
                    
                tracking = app.get("email_tracking", {})
                if not isinstance(tracking, dict):
                    logger.warning(f"Invalid email_tracking format in campaign {campaign_id}")
                    continue
                
                # Initial emails
                email1 = tracking.get("email1", {})
                if isinstance(email1, dict) and email1.get("status") == EmailStatus.SENT.value:
                    email_sequence_stats["initial"] += 1
                    if tracking.get("overall_status") == EmailStatus.REPLIED.value:
                        email_sequence_replies["initial"] += 1
                
                # Follow-up emails
                for i in range(1, 4):
                    follow_key = f"follow{i}"
                    follow_email = tracking.get(follow_key, {})
                    if isinstance(follow_email, dict) and follow_email.get("status") == EmailStatus.SENT.value:
                        email_sequence_stats[f"followup{i}"] += 1
                        if tracking.get("overall_status") == EmailStatus.REPLIED.value:
                            email_sequence_replies[f"followup{i}"] += 1
                
                # Email status stats
                status = tracking.get("overall_status")
                if status == EmailStatus.PENDING.value:
                    email_status_stats["pending"] += 1
                elif status == EmailStatus.SENT.value:
                    email_status_stats["sent"] += 1
                elif status == EmailStatus.REPLIED.value:
                    email_status_stats["replied"] += 1
                elif status == EmailStatus.FAILED.value:
                    email_status_stats["failed"] += 1
        except Exception as e: 
            logger.error(f"Analytics error for campaign {campaign_id}: {e}")
    
    # Calculate overall rates
    if total_stats["initial_emails_sent"] > 0: 
        total_stats["overall_reply_rate"] = round((total_stats["total_replies"] / total_stats["initial_emails_sent"]) * 100, 2)
    else: 
        total_stats["overall_reply_rate"] = 0
    
    if total_stats["apps_with_contacts"] > 0: 
        total_stats["overall_completion_rate"] = round((total_stats["apps_completed"] / total_stats["apps_with_contacts"]) * 100, 2)
    else: 
        total_stats["overall_completion_rate"] = 0
    
    # Calculate growth percentages compared to previous period
    prev = historical_data["previous_period"]
    growth = {
        "campaign_growth": calculate_growth(total_stats["total_campaigns"], prev["total_campaigns"]),
        "email_growth": calculate_growth(total_stats["initial_emails_sent"], prev["total_emails_sent"]),
        "reply_rate_growth": calculate_growth(total_stats["overall_reply_rate"], 
                                           (prev["total_replies"] / prev["total_emails_sent"] * 100) if prev["total_emails_sent"] > 0 else 0),
        "app_growth": calculate_growth(total_stats["total_apps"], prev["total_apps"])
    }
    
    # Prepare the response
    response = {
        "overall_summary": {
            **dict(total_stats),
            "growth": growth
        },
        "email_sequence": dict(email_sequence_stats),
        "email_sequence_replies": dict(email_sequence_replies),
        "email_status": dict(email_status_stats),
        "campaign_breakdown": campaign_details
    }
    
    return jsonify(response)

def calculate_growth(current, previous):
    """Calculate percentage growth between current and previous values"""
    if previous == 0:
        return 100 if current > 0 else 0
    return round(((current - previous) / previous) * 100, 1)

# ==============================================================================
# 6. APP STARTUP & ERROR HANDLERS
# ==============================================================================

@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found"}), 404

if __name__ == '__main__':
    print("🚀 UndrAds Advanced Email Campaign System v5.4 (Thread-Safety Fix)")
    print("-------------------------------------------------")
    print("Recommended Workflow:")
    print("1. POST /campaign/upload              -> Upload JSON data")
    print("2. POST /campaign/extract-contacts    -> Extract and initialize contacts")
    print("3. POST /campaign/generate-preview    -> (Optional) Preview AI email copy")
    print("4. POST /campaign/generate-emails     -> Generate all email copy with AI")
    print("5. POST /campaign/send-emails         -> Start the email sending sequence")
    print("-------------------------------------------------")
    print(f"💾 Data directory: {os.path.abspath(DATA_DIR)}")
    print("🔒 Ensure GMAIL_USER and GMAIL_PASSWORD (as App Password) are set.")
    print("⚠️ WARNING: Follow-ups use non-persistent timers. A server restart will lose scheduled tasks.")
    app.run(debug=True, host='0.0.0.0', port=8081)