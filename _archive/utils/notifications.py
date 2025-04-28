import requests
import os
import logging

logger = logging.getLogger(__name__)

def send_telegram_message(message, token=None, chat_id=None):
    """
    Send a message via Telegram bot.
    
    Args:
        message (str): The message to send
        token (str, optional): Bot token. Defaults to environment variable.
        chat_id (str, optional): Chat ID. Defaults to environment variable.
        
    Returns:
        bool: Success status
    """
    token = token or os.environ.get('TELEGRAM_BOT_TOKEN')
    chat_id = chat_id or os.environ.get('TELEGRAM_CHAT_ID')
    
    if not token or not chat_id:
        logger.info("Telegram notification skipped: missing token or chat ID")
        return False
    
    try:
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        payload = {
            "chat_id": chat_id,
            "text": message,
            "parse_mode": "Markdown"
        }
        response = requests.post(url, data=payload, timeout=10)
        success = response.status_code == 200
        
        if success:
            logger.info("Telegram notification sent successfully")
        else:
            logger.warning(f"Telegram notification failed with status code {response.status_code}")
            
        return success
    except Exception as e:
        logger.error(f"Failed to send Telegram notification: {e}")
        return False 