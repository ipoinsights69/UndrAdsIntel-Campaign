# Email Campaign Automation System

A Flask-based system for automating email campaigns with AI-powered email generation and automated follow-ups.

## Core Features

- AI-powered email generation
- Automated follow-up scheduling
- Contact extraction and validation
- Campaign progress tracking
- Email response monitoring
- Analytics and reporting

## API Documentation

### 1. Upload Campaign
**Endpoint:** `/campaign/upload`
**Method:** POST

Upload a new campaign data file.

**Request:**
- Form Data:
  - `campaign_id` (optional): Custom ID for the campaign
  - `file`: JSON file containing campaign data

**Response:**
```json
{
  "message": "Campaign uploaded successfully",
  "campaign_id": "[campaign_id]",
  "total_apps": [number]
}
```

### 2. Extract Contacts
**Endpoint:** `/campaign/extract-contacts`
**Method:** POST

Extract email contacts from campaign data.

**Request:**
```json
{
  "campaign_id": "[campaign_id]"
}
```

### 3. Generate Email Preview
**Endpoint:** `/campaign/generate-preview`
**Method:** POST

Generate preview of AI-generated emails for selected apps.

**Request:**
```json
{
  "campaign_id": "[campaign_id]",
  "app_indices": [array_of_indices],
  "selected_keys": [array_of_keys],
  "tone": "conversational",
  "num_followups": 3
}
```

### 4. Generate Emails
**Endpoint:** `/campaign/generate-emails`
**Method:** POST

Start background task for email generation.

**Request:**
```json
{
  "campaign_id": "[campaign_id]",
  "limit": [optional_number],
  "tone": "conversational",
  "num_followups": 3,
  "selected_keys": [array_of_keys]
}
```

### 5. Send Emails
**Endpoint:** `/campaign/send-emails`
**Method:** POST

Start email sending campaign.

**Request:**
```json
{
  "campaign_id": "[campaign_id]",
  "followup_delay_hours": [number],
  "followup_delay_minutes": [number],
  "max_followups": 3,
  "test_mode": true,
  "sender_name": "The UndrAds Team",
  "sender_company": "UndrAds",
  "limit": [optional_number]
}
```

### 6. Campaign Progress
**Endpoint:** `/campaign/progress`
**Method:** GET

Get campaign progress and statistics.

**Query Parameters:**
- `campaign_id`: Campaign identifier

### 7. Get Campaign Apps Status
**Endpoint:** `/campaign/<campaign_id>/apps`
**Method:** GET

Get detailed status for all apps in a campaign.

### 8. List Campaigns
**Endpoint:** `/campaigns`
**Method:** GET

List all campaigns with their progress and statistics.

### 9. Overall Analytics
**Endpoint:** `/analytics/overall`
**Method:** GET

Get aggregated analytics across all campaigns.

### 10. Get Common Keys
**Endpoint:** `/campaign/<campaign_id>/common-keys`
**Method:** GET

Analyze campaign data to find most common top-level keys for AI generation.

## Configuration

### Environment Variables

- `GMAIL_USER`: Gmail account username
- `GMAIL_PASSWORD`: Gmail account password or app-specific password

## Project Structure

```
├── main.py           # Main application file
├── campaign_data/    # Campaign data storage
│   └── [campaign_id]/
│       ├── raw_data.json     # Original uploaded data
│       ├── campaign_data.json # Processed campaign data
│       └── progress.json     # Campaign progress tracking
└── tasks.db         # Background tasks database
```

## Best Practices

1. Always use test mode first when sending emails
2. Monitor campaign progress regularly
3. Check for email responses before sending follow-ups
4. Keep campaign data organized by campaign ID
5. Use appropriate delay between follow-ups

## Error Handling

The API uses standard HTTP status codes:
- 200: Success
- 202: Accepted (for background tasks)
- 400: Bad Request
- 404: Not Found
- 409: Conflict (task already running)
- 500: Internal Server Error

## Thread Safety

The system implements thread-safe operations for:
- Campaign data access
- Progress tracking
- Email sending
- Background task management