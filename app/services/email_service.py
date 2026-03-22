import logging
from typing import Optional

logger = logging.getLogger(__name__)


class EmailService:
    @staticmethod
    async def send_reset_password_email(email: str, token: str):
        """
        Mock email service that logs the reset link to the console.
        In a real application, this would send an actual email via SMTP or a provider like SendGrid.
        """
        reset_link = f"lsc://reset-password?token={token}"
        # Also providing a web link example
        web_reset_link = f"http://localhost:5173/reset-password?token={token}"
        
        email_content = f"""
        Subject: LSC Translator - Password Reset Request
        To: {email}
        
        You requested a password reset for your LSC Translator account.
        Please click the link below or use the token in the app:
        
        Token: {token}
        App Link: {reset_link}
        Web Link: {web_reset_link}
        
        This link and token will expire in 1 hour.
        If you did not request this, please ignore this email.
        """
        
        logger.info(f"Sending reset password email to {email}")
        print("------------------------------------------")
        print(email_content)
        print("------------------------------------------")
        
        # Simulating async work
        return True

    @staticmethod
    async def send_verification_email(email: str, token: str):
        """
        Mock email service for email verification.
        """
        verify_link = f"lsc://verify-email?token={token}"
        web_verify_link = f"http://localhost:8000/api/auth/verify-email?token={token}"
        
        email_content = f"""
        Subject: LSC Translator - Verify Your Email
        To: {email}
        
        Welcome to LSC Translator!
        Please click the link below to verify your email address:
        
        Token: {token}
        App Link: {verify_link}
        Web Link: {web_verify_link}
        
        This link and token will expire in 24 hours.
        """
        
        logger.info(f"Sending verification email to {email}")
        print("------------------------------------------")
        print(email_content)
        print("------------------------------------------")
        
        return True


email_service = EmailService()
