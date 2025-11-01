"""Email utilities for authentication."""

import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

from kohakuboard_server.config import cfg
from kohakuboard_server.logger import logger_api


def send_verification_email(to_email: str, username: str, token: str) -> bool:
    """Send email verification email.

    Args:
        to_email: Recipient email
        username: Username
        token: Verification token

    Returns:
        True if email sent successfully, False otherwise
    """
    if not cfg.smtp.enabled:
        verify_link = f"{cfg.app.base_url}/api/auth/verify-email?token={token}"
        logger_api.info(f"SMTP disabled. Verification link: {verify_link}")
        return True

    subject = "Verify your KohakuBoard account"
    base_url = cfg.app.base_url or "http://localhost:28081"
    verify_link = f"{base_url}/api/auth/verify-email?token={token}"

    # Plain text version
    text_body = f"""
Hello {username},

Please verify your email address by clicking the link below:

{verify_link}

This link will expire in 24 hours.

If you didn't create this account, please ignore this email.

Best regards,
KohakuBoard Team
"""

    # HTML version
    html_body = f"""
<!DOCTYPE html>
<html>
<body style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto; padding: 20px;">
    <h2 style="color: #667eea;">Welcome to KohakuBoard!</h2>
    <p>Hello <strong>{username}</strong>,</p>
    <p>Please verify your email address to activate your account:</p>
    <p style="text-align: center; margin: 30px 0;">
        <a href="{verify_link}"
           style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                  color: white;
                  padding: 12px 30px;
                  text-decoration: none;
                  border-radius: 5px;
                  display: inline-block;">
            Verify Email Address
        </a>
    </p>
    <p style="color: #666;">Or copy this link: <code>{verify_link}</code></p>
    <p style="color: #999; font-size: 12px;">This link expires in 24 hours.</p>
</body>
</html>
"""

    try:
        msg = MIMEMultipart("alternative")
        msg["From"] = cfg.smtp.from_email
        msg["To"] = to_email
        msg["Subject"] = subject

        msg.attach(MIMEText(text_body, "plain"))
        msg.attach(MIMEText(html_body, "html"))

        with smtplib.SMTP(cfg.smtp.host, cfg.smtp.port) as server:
            if cfg.smtp.use_tls:
                server.starttls()
            if cfg.smtp.username and cfg.smtp.password:
                server.login(cfg.smtp.username, cfg.smtp.password)
            server.send_message(msg)

        logger_api.success(f"Verification email sent to {to_email}")
        return True
    except Exception as e:
        logger_api.error(f"Failed to send verification email: {e}")
        return False
