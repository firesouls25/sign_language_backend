from google.cloud import storage
from app.config import settings
import logging
from typing import Optional
import uuid

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

    async def upload_file(self, file_data: bytes, filename: str, content_type: str) -> Optional[str]:
        if not self.bucket:
            return None
        
        try:
            blob = self.bucket.blob(f"{uuid.uuid4()}/{filename}")
            blob.upload_from_string(file_data, content_type=content_type)
            return blob.public_url
        except Exception as e:
            logger.error(f"Upload error: {e}")
            return None

    async def delete_file(self, url: str) -> bool:
        if not self.bucket:
            return False
        
        try:
            blob_name = url.split(f"/{self.bucket_name}/")[-1]
            blob = self.bucket.blob(blob_name)
            blob.delete()
            return True
        except Exception as e:
            logger.error(f"Delete error: {e}")
            return False


storage_service = None


def get_storage_service() -> StorageService:
    global storage_service
    if storage_service is None:
        storage_service = StorageService()
    return storage_service
