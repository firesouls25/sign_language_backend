from fastapi import APIRouter, Depends, HTTPException, status, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Optional
from app.database import get_db
from app.schemas.user import (
    UserCreate,
    UserLogin,
    Token,
    UserResponse,
    ForgotPasswordRequest,
    ResetPasswordRequest,
)
from app.services.auth_service import AuthService
from app.services import oauth_service

router = APIRouter(prefix="/api/auth", tags=["auth"])
security = HTTPBearer()


@router.get("/login/{provider}")
async def login_with_oauth(provider: str, request: Request):
    if provider not in ["google", "apple"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Unsupported OAuth provider"
        )

    redirect_uri = oauth_service.get_oauth_redirect_url(provider)

    if provider == "google":
        return await oauth_service.oauth.google.authorize_redirect(
            request,
            redirect_uri=redirect_uri,
        )
    elif provider == "apple":
        return await oauth_service.oauth.apple.authorize_redirect(
            request,
            redirect_uri=redirect_uri,
        )


@router.get("/callback/{provider}", response_model=Token)
async def oauth_callback(
    provider: str,
    request: Request,
    code: Optional[str] = None,
    error: Optional[str] = None,
    db: AsyncSession = Depends(get_db),
):
    if error:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=f"OAuth error: {error}"
        )

    if not code:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Missing authorization code"
        )

    redirect_uri = oauth_service.get_oauth_redirect_url(provider)

    try:
        user = None
        if provider == "google":
            token = await oauth_service.oauth.google.fetch_access_token(
                code=code, redirect_uri=redirect_uri
            )
            userinfo = await oauth_service.get_google_userinfo(token["access_token"])

            user = await oauth_service.find_or_create_oauth_user(
                db=db,
                provider="google",
                provider_id=userinfo["sub"],
                email=userinfo["email"],
                full_name=userinfo.get("name"),
                avatar_url=userinfo.get("picture"),
            )

        elif provider == "apple":
            token = await oauth_service.oauth.apple.fetch_access_token(
                code=code, redirect_uri=redirect_uri
            )
            userinfo = await oauth_service.get_apple_userinfo(token["access_token"])

            user = await oauth_service.find_or_create_oauth_user(
                db=db,
                provider="apple",
                provider_id=userinfo["sub"],
                email=userinfo["email"],
            )

        if not user:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Failed to create or find user",
            )

        return await oauth_service.create_tokens_for_user(user)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Authentication failed: {str(e)}",
        )


@router.get("/callback-webview/{provider}")
async def oauth_callback_webview(
    provider: str,
    code: Optional[str] = None,
    error: Optional[str] = None,
    db: AsyncSession = Depends(get_db),
):
    if error:
        return f"<html><script>window.location.href='lsc://oauth/callback?error={error}';</script></html>"

    if not code:
        return "<html><script>window.location.href='lsc://oauth/callback?error=missing_code';</script></html>"

    redirect_uri = oauth_service.get_oauth_redirect_url(provider)

    try:
        if provider == "google":
            token = await oauth_service.oauth.google.fetch_access_token(
                code=code, redirect_uri=redirect_uri
            )
            userinfo = await oauth_service.get_google_userinfo(token["access_token"])

            user = await oauth_service.find_or_create_oauth_user(
                db=db,
                provider="google",
                provider_id=userinfo["sub"],
                email=userinfo["email"],
                full_name=userinfo.get("name"),
                avatar_url=userinfo.get("picture"),
            )

        elif provider == "apple":
            token = await oauth_service.oauth.apple.fetch_access_token(
                code=code, redirect_uri=redirect_uri
            )
            userinfo = await oauth_service.get_apple_userinfo(token["access_token"])

            user = await oauth_service.find_or_create_oauth_user(
                db=db,
                provider="apple",
                provider_id=userinfo["sub"],
                email=userinfo["email"],
            )

        if not user:
            return "<html><script>window.location.href='lsc://oauth/callback?error=user_creation_failed';</script></html>"

        tokens = await oauth_service.create_tokens_for_user(user)

        return f"<html><script>window.location.href='lsc://oauth/callback?access_token={tokens['access_token']}&refresh_token={tokens['refresh_token']}&token_type=bearer';</script></html>"

    except Exception as e:
        return f"<html><script>window.location.href='lsc://oauth/callback?error={str(e)}';</script></html>"


async def oauth_callback_dev(
    provider: str,
    code: Optional[str] = None,
    error: Optional[str] = None,
    db: AsyncSession = Depends(get_db),
):
    if error:
        return f"<html><body><h1>Error: {error}</h1><p>You can close this window and return to the app.</p></body></html>"

    if not code:
        return "<html><body><h1>Missing code</h1><p>You can close this window and return to the app.</p></body></html>"

    redirect_uri = oauth_service.get_oauth_redirect_url(provider)

    try:
        if provider == "google":
            token = await oauth_service.oauth.google.fetch_access_token(
                code=code, redirect_uri=redirect_uri
            )
            userinfo = await oauth_service.get_google_userinfo(token["access_token"])

            user = await oauth_service.find_or_create_oauth_user(
                db=db,
                provider="google",
                provider_id=userinfo["sub"],
                email=userinfo["email"],
                full_name=userinfo.get("name"),
                avatar_url=userinfo.get("picture"),
            )

        elif provider == "apple":
            token = await oauth_service.oauth.apple.fetch_access_token(
                code=code, redirect_uri=redirect_uri
            )
            userinfo = await oauth_service.get_apple_userinfo(token["access_token"])

            user = await oauth_service.find_or_create_oauth_user(
                db=db,
                provider="apple",
                provider_id=userinfo["sub"],
                email=userinfo["email"],
            )

        if not user:
            return "<html><body><h1>Failed to create user</h1><p>You can close this window and return to the app.</p></body></html>"

        tokens = await oauth_service.create_tokens_for_user(user)

        return f"""
        <html>
        <head>
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Login Successful</title>
            <style>
                body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; padding: 20px; max-width: 600px; margin: 0 auto; }}
                .token-box {{ background: #f5f5f5; padding: 15px; border-radius: 8px; margin: 10px 0; word-break: break-all; }}
                .label {{ font-weight: bold; color: #333; }}
                button {{ background: #673AB7; color: white; border: none; padding: 12px 24px; border-radius: 8px; cursor: pointer; font-size: 16px; }}
                button:hover {{ background: #512DA8; }}
            </style>
        </head>
        <body>
            <h1>✅ Login Successful!</h1>
            <p>Copy these tokens and paste them in the app:</p>
            <div class="token-box">
                <div class="label">access_token:</div>
                {tokens["access_token"]}
            </div>
            <div class="token-box">
                <div class="label">refresh_token:</div>
                {tokens["refresh_token"]}
            </div>
            <button onclick="copyTokens()">Copy All Tokens</button>
            <script>
                function copyTokens() {{
                    const text = `access_token={tokens["access_token"]}&refresh_token={tokens["refresh_token"]}`;
                    navigator.clipboard.writeText(text).then(() => {{
                        alert('Tokens copied! You can now paste them in the app.');
                    }});
                }}
            </script>
            <p><small>You can close this window after copying the tokens.</small></p>
        </body>
        </html>
        """

    except Exception as e:
        return f"<html><body><h1>Error: {str(e)}</h1><p>You can close this window and return to the app.</p></body></html>"


@router.post(
    "/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED
)
async def register(user_data: UserCreate, db: AsyncSession = Depends(get_db)):
    user = await AuthService.register(db, user_data)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email or username already registered",
        )
    return user


@router.post("/login", response_model=Token)
async def login(credentials: UserLogin, db: AsyncSession = Depends(get_db)):
    token = await AuthService.login(db, credentials)
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials"
        )
    return token


@router.post("/refresh", response_model=Token)
async def refresh_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    new_access_token = await AuthService.refresh_access_token(credentials.credentials)
    if not new_access_token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid refresh token"
        )
    return Token(
        access_token=new_access_token,
        refresh_token=credentials.credentials,
    )


@router.get("/me", response_model=UserResponse)
async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: AsyncSession = Depends(get_db),
):
    user_id = await AuthService.get_user_from_token(credentials.credentials)
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token"
        )

    user = await AuthService.get_user_by_id(db, user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="User not found"
        )
    return user


@router.post("/logout")
async def logout(
    request: Request,
    credentials: HTTPAuthorizationCredentials = Depends(security),
):
    # We might want to pass the refresh token too if the client sends it
    # For now, let's try to get it from the body if possible, or just blacklist the access token
    access_token = credentials.credentials
    
    # Optional: try to get refresh token from request body
    refresh_token = None
    try:
        body = await request.json()
        refresh_token = body.get("refresh_token")
    except:
        pass

    await AuthService.logout(access_token, refresh_token)
    return {"message": "Successfully logged out"}


@router.post("/forgot-password")
async def forgot_password(
    request: ForgotPasswordRequest, db: AsyncSession = Depends(get_db)
):
    success = await AuthService.forgot_password(db, request.email)
    # Always return success message for security (don't reveal if email exists)
    return {
        "message": "If the email is registered, you will receive a reset link shortly."
    }


@router.post("/reset-password")
async def reset_password(
    request: ResetPasswordRequest, db: AsyncSession = Depends(get_db)
):
    success = await AuthService.reset_password(db, request.token, request.new_password)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid or expired token"
        )
    return {"message": "Password successfully reset"}
