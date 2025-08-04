import os
import gdown
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

def download_model_from_drive(drive_url, model_path):
    """
    Tải model từ Google Drive với URL public
    """
    try:
        # Tạo thư mục models nếu chưa có
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        # Kiểm tra nếu model đã tồn tại
        if os.path.exists(model_path):
            logger.info(f"✅ Model already exists at: {model_path}")
            return True
        
        logger.info(f"🔄 Downloading model from Google Drive...")
        logger.info(f"📁 Target path: {model_path}")
        
        # Trích xuất file ID từ Google Drive URL
        if "drive.google.com" in drive_url:
            if "/file/d/" in drive_url:
                file_id = drive_url.split("/file/d/")[1].split("/")[0]
            elif "id=" in drive_url:
                file_id = drive_url.split("id=")[1].split("&")[0]
            else:
                raise ValueError("Invalid Google Drive URL format")
        else:
            raise ValueError("URL is not a Google Drive link")
        
        # Tạo direct download URL
        download_url = f"https://drive.google.com/uc?id={file_id}"
        
        # Tải file
        gdown.download(download_url, model_path, quiet=False)
        
        # Kiểm tra file đã tải thành công
        if os.path.exists(model_path) and os.path.getsize(model_path) > 0:
            file_size = os.path.getsize(model_path) / (1024 * 1024)  # MB
            logger.info(f"✅ Model downloaded successfully! Size: {file_size:.2f}MB")
            return True
        else:
            logger.error("❌ Downloaded file is empty or corrupted")
            if os.path.exists(model_path):
                os.remove(model_path)
            return False
            
    except Exception as e:
        logger.error(f"❌ Error downloading model: {str(e)}")
        # Xóa file lỗi nếu có
        if os.path.exists(model_path):
            try:
                os.remove(model_path)
            except:
                pass
        return False

def ensure_model_available(drive_url, model_path):
    """
    Đảm bảo model có sẵn, tải từ Drive nếu cần
    """
    if not os.path.exists(model_path):
        logger.info("🔍 Model not found locally, downloading from Google Drive...")
        return download_model_from_drive(drive_url, model_path)
    else:
        # Kiểm tra kích thước file
        file_size = os.path.getsize(model_path)
        if file_size < 1024 * 1024:  # Nhỏ hơn 1MB có thể là file lỗi
            logger.warning("⚠️ Model file seems too small, re-downloading...")
            os.remove(model_path)
            return download_model_from_drive(drive_url, model_path)
        else:
            logger.info(f"✅ Model found locally: {model_path}")
            return True
