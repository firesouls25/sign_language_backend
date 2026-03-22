import logging
import json
import time
import uuid
from typing import Any, Dict
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from contextvars import ContextVar

# Context variable to store request ID
request_id_ctx_var: ContextVar[str] = ContextVar("request_id", default="system")


class JSONFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        log_record: Dict[str, Any] = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "message": record.getMessage(),
            "logger": record.name,
            "request_id": request_id_ctx_var.get(),
        }
        
        # Add extra fields if they exist
        extra_info = getattr(record, "extra_info", {})
        if extra_info:
            log_record.update(extra_info)
            
        if record.exc_info:
            log_record["exception"] = self.formatException(record.exc_info)
            
        return json.dumps(log_record)


def setup_logging():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # Remove existing handlers
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        
    # JSON Handler
    handler = logging.StreamHandler()
    handler.setFormatter(JSONFormatter())
    logger.addHandler(handler)
    
    # Silence third-party loggers if needed
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)


class LoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        request_id = str(uuid.uuid4())
        token = request_id_ctx_var.set(request_id)
        
        start_time = time.time()
        
        logger = logging.getLogger("app.middleware")
        
        # Log request
        logger.info(
            f"Incoming request: {request.method} {request.url.path}",
            extra={"extra_info": {
                "method": request.method,
                "path": request.url.path,
                "client_host": request.client.host if request.client else "unknown",
            }}
        )
        
        try:
            response = await call_next(request)
            
            process_time = time.time() - start_time
            
            # Log response
            logger.info(
                f"Outgoing response: {request.method} {request.url.path} - {response.status_code}",
                extra={"extra_info": {
                    "method": request.method,
                    "path": request.url.path,
                    "status_code": response.status_code,
                    "process_time_ms": float(f"{process_time * 1000:.2f}"),
                }}
            )
            
            # Add request ID to response header
            response.headers["X-Request-ID"] = request_id
            
            return response
            
        except Exception as e:
            process_time = time.time() - start_time
            logger.error(
                f"Request failed: {request.method} {request.url.path} - {str(e)}",
                exc_info=True,
                extra={"extra_info": {
                    "method": request.method,
                    "path": request.url.path,
                    "process_time_ms": float(f"{process_time * 1000:.2f}"),
                }}
            )
            raise e
        finally:
            request_id_ctx_var.reset(token)


# Helper to log with extra info
def get_logger(name: str):
    return logging.getLogger(name)
