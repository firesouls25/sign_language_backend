from google.cloud import storage
from app.config import settings
import logging
from typing import Optional
import uuid
import os
from pathlib import Path

logger = logging.getLogger(__name__)


class StorageService:
    def __init__(self):
        self.bucket_name = settings.GCS_BUCKET_NAME
        self.client = None
        self.bucket = None
        
        if settings.GOOGLE_APPLICATION_CREDENTIALS:
            try:
                self.client = storage.Client()
                self.bucket = self.client.bucket(self.bucket_name)
            except Exception as e:
                logger.warning(f"Could not initialize GCS: {e}")
        
        # Ensure local upload dir exists
        self.upload_dir = settings.UPLOAD_DIR
        os.makedirs(self.upload_dir, exist_ok=True)

    async def upload_file(self, file_data: bytes, filename: str, content_type: str) -> Optional[str]:
        try:
            unique_filename = f"{uuid.uuid4()}_{filename}"
            
            if self.bucket:
                blob = self.bucket.blob(unique_filename)
                blob.upload_from_string(file_data, content_type=content_type)
                return blob.public_url
            else:
                # Local storage fallback
                file_path = os.path.join(self.upload_dir, unique_filename)
                with open(file_path, "wb") as f:
                    f.write(file_data)
                
                # Return a relative URL that will be served by FastAPI
                return f"/api/uploads/{unique_filename}"
                
        except Exception as e:
            logger.error(f"Upload error: {e}")
            return None

    async def delete_file(self, url: str) -> bool:
        try:
            if self.bucket and f"/{self.bucket_name}/" in url:
                blob_name = url.split(f"/{self.bucket_name}/")[-1]
                blob = self.bucket.blob(blob_name)
                blob.delete()
                return True
            elif "/api/uploads/" in url:
                # Local storage delete
                filename = url.split("/api/uploads/")[-1]
                file_path = os.path.join(self.upload_dir, filename)
                if os.path.exists(file_path):
                    os.remove(file_path)
                    return True
            return False
        except Exception as e:
            logger.error(f"Delete error: {e}")
            return False


storage_service = None


def get_storage_service() -> StorageService:
    global storage_service
    if storage_service is None:
        storage_service = StorageService()
    return storage_service
