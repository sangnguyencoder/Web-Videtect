import os
import json
import cv2
import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import base64
from datetime import datetime
import logging
import threading
import time
from functools import wraps

# Import từ project hiện tại
from src.model import build_model
from src.utils import video_to_tensor, FRAMES_PER_CLIP, IMG_SIZE
from src.model_loader import ensure_model_available

# Configure logging với format đẹp hơn
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['SECRET_KEY'] = 'your-secret-key-here-change-in-production'

# Tạo thư mục uploads nếu chưa có
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Model configuration
MODEL_PATH = os.path.join('models', 'violence_model.h5')
DRIVE_MODEL_URL = 'https://drive.google.com/file/d/1ZrDA5PhjgQvE9pg5q_y_9ZvHviDtm_1j/view?usp=drive_link'

model = None
model_loaded = False
model_load_time = None

def load_model():
    """Load the violence detection model with error handling"""
    global model, model_loaded, model_load_time
    try:
        start_time = time.time()
        logger.info("🔄 Loading violence detection model...")
        
        # Đảm bảo model có sẵn (tải từ Drive nếu cần)
        if not ensure_model_available(DRIVE_MODEL_URL, MODEL_PATH):
            logger.error("❌ Failed to download or locate model file")
            model_loaded = False
            return
        
        # Xây dựng model architecture
        model = build_model()
        
        # Load weights
        model.load_weights(MODEL_PATH)
        model_loaded = True
        model_load_time = time.time() - start_time
        
        logger.info(f"✅ Model loaded successfully in {model_load_time:.2f}s!")
        logger.info(f"📊 Model architecture: 3D-CNN + LSTM")
        logger.info(f"🎬 Input shape: ({FRAMES_PER_CLIP}, {IMG_SIZE}, {IMG_SIZE}, 3)")
        
    except Exception as e:
        logger.error(f"❌ Error loading model: {e}")
        model_loaded = False

# Load model khi khởi động trong thread riêng
def load_model_async():
    load_model()

model_thread = threading.Thread(target=load_model_async)
model_thread.daemon = True
model_thread.start()

# Cấu hình cho phép upload
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'webm', 'flv'}

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def require_model_loaded(f):
    """Decorator to check if model is loaded"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not model_loaded:
            return jsonify({
                'error': 'AI model is not ready. Please wait a moment and try again.',
                'status': 'model_not_ready',
                'retry_after': 5
            }), 503
        return f(*args, **kwargs)
    return decorated_function

@app.route('/')
def index():
    """Render main page"""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
@require_model_loaded
def upload_video():
    """Handle video upload and analysis with enhanced error handling"""
    
    # Validate request
    if 'video' not in request.files:
        return jsonify({
            'error': 'No video file found in request',
            'code': 'NO_FILE'
        }), 400

    file = request.files['video']
    if file.filename == '':
        return jsonify({
            'error': 'No file selected',
            'code': 'EMPTY_FILENAME'
        }), 400

    if not allowed_file(file.filename):
        return jsonify({
            'error': 'Unsupported file format. Please upload MP4, AVI, MOV, MKV, WEBM or FLV files',
            'supported_formats': list(ALLOWED_EXTENSIONS),
            'code': 'INVALID_FORMAT'
        }), 400

    # Check file size
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)
    
    if file_size > app.config['MAX_CONTENT_LENGTH']:
        return jsonify({
            'error': f'File too large ({file_size / (1024*1024):.1f}MB). Maximum allowed size is 100MB.',
            'max_size_mb': 100,
            'file_size_mb': round(file_size / (1024*1024), 1),
            'code': 'FILE_TOO_LARGE'
        }), 413

    if file_size == 0:
        return jsonify({
            'error': 'File appears to be empty. Please select a valid video file.',
            'code': 'EMPTY_FILE'
        }), 400

    try:
        # Generate unique filename
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        unique_filename = f"{timestamp}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        
        logger.info(f"📁 Saving uploaded file: {unique_filename} ({file_size / (1024*1024):.1f}MB)")
        
        # Save file with progress tracking
        start_time = time.time()
        file.save(filepath)
        save_time = time.time() - start_time
        
        # Verify file was saved correctly
        if not os.path.exists(filepath) or os.path.getsize(filepath) != file_size:
            return jsonify({
                'error': 'File save failed. Please try again.',
                'code': 'SAVE_FAILED'
            }), 500

        logger.info(f"💾 File saved in {save_time:.2f}s")

        # Process video
        logger.info(f"🔍 Starting video analysis: {unique_filename}")
        analysis_start = time.time()
        
        result = process_video(filepath)
        
        analysis_time = time.time() - analysis_start
        result['processing_time'] = round(analysis_time, 2)
        result['file_info'] = {
            'original_name': filename,
            'size_mb': round(file_size / (1024*1024), 2),
            'save_time': round(save_time, 2)
        }
        
        logger.info(f"✅ Analysis completed in {analysis_time:.2f}s")

        # Clean up file
        try:
            os.remove(filepath)
            logger.info(f"🗑️ Temporary file removed: {unique_filename}")
        except Exception as e:
            logger.warning(f"⚠️ Could not remove temporary file {unique_filename}: {e}")

        return jsonify(result)

    except Exception as e:
        logger.error(f"❌ Error processing video upload: {str(e)}")
        
        # Clean up file if exists
        if 'filepath' in locals() and os.path.exists(filepath):
            try:
                os.remove(filepath)
            except:
                pass
                
        # Return appropriate error message
        error_message = str(e)
        if "out of memory" in error_message.lower():
            error_message = "Video too complex to process. Please try a shorter or lower resolution video."
        elif "corrupted" in error_message.lower():
            error_message = "Video file appears to be corrupted. Please try a different file."
        
        return jsonify({
            'error': f'Error processing video: {error_message}',
            'code': 'PROCESSING_ERROR'
        }), 500

def process_video(video_path):
    """Process video and return prediction results with enhanced error handling"""
    try:
        logger.info(f"🎬 Starting video analysis: {video_path}")
        
        # Get video info first
        video_info = get_video_info(video_path)
        
        # Validate video
        if video_info['duration'] == 0:
            raise ValueError("Video appears to be corrupted or unreadable")
        
        if video_info['duration'] > 300:  # 5 minutes
            raise ValueError("Video too long. Please use videos shorter than 5 minutes")
        
        # Convert video to tensor
        logger.info("🔄 Converting video to tensor...")
        tensor_start = time.time()
        video_tensor = video_to_tensor(video_path)
        tensor_time = time.time() - tensor_start
        
        if video_tensor is None:
            raise ValueError("Failed to process video frames")
        
        logger.info(f"✅ Tensor conversion completed in {tensor_time:.2f}s")
        logger.info(f"📊 Tensor shape: {video_tensor.shape}")
        
        # Add batch dimension
        video_tensor = video_tensor[None, ...]
        
        # Run prediction
        logger.info("🧠 Running AI prediction...")
        pred_start = time.time()
        prediction = model.predict(video_tensor, verbose=0)
        pred_time = time.time() - pred_start
        
        probability = float(prediction[0][0])
        
        # Determine label and confidence
        threshold = 0.5
        label = "Non-Violence" if probability > threshold else "Violence"
        confidence = probability if label == "Non-Violence" else 1 - probability
        
        # Create comprehensive result
        result = {
            'label': label,
            'confidence': round(confidence * 100, 2),
            'probability': round(probability * 100, 2),
            'threshold': threshold * 100,
            'timestamp': datetime.now().isoformat(),
            'video_info': video_info,
            'model_info': {
                'frames_per_clip': FRAMES_PER_CLIP,
                'img_size': IMG_SIZE,
                'architecture': '3D-CNN + LSTM',
                'version': '2.0'
            },
            'performance': {
                'tensor_conversion_time': round(tensor_time, 2),
                'prediction_time': round(pred_time, 2),
                'total_processing_time': round(tensor_time + pred_time, 2)
            }
        }
        
        logger.info(f"🎯 Analysis result: {label} ({confidence*100:.2f}% confidence)")
        logger.info(f"⚡ Prediction completed in {pred_time:.2f}s")
        
        return result
    
    except Exception as e:
        logger.error(f"❌ Error in process_video: {str(e)}")
        raise Exception(f"Video analysis error: {str(e)}")

def get_video_info(video_path):
    """Extract comprehensive video information"""
    try:
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError("Cannot open video file")
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Calculate duration
        duration = frame_count / fps if fps > 0 else 0
        
        # Get codec information
        fourcc = cap.get(cv2.CAP_PROP_FOURCC)
        codec = "".join([chr((int(fourcc) >> 8 * i) & 0xFF) for i in range(4)])
        
        cap.release()
        
        return {
            'fps': round(fps, 2),
            'frame_count': frame_count,
            'duration': round(duration, 2),
            'resolution': f"{width}x{height}",
            'width': width,
            'height': height,
            'codec': codec.strip(),
            'aspect_ratio': round(width / height, 2) if height > 0 else 0
        }
        
    except Exception as e:
        logger.error(f"❌ Error getting video info: {str(e)}")
        return {
            'fps': 0,
            'frame_count': 0,
            'duration': 0,
            'resolution': "Unknown",
            'width': 0,
            'height': 0,
            'codec': "Unknown",
            'aspect_ratio': 0
        }

@app.route('/webcam', methods=['POST'])
@require_model_loaded
def webcam_analysis():
    """Process frame from webcam for real-time analysis with enhanced error handling"""
    
    try:
        # Validate request
        data = request.get_json()
        if not data or 'frame' not in data:
            return jsonify({
                'error': 'No frame data found in request',
                'code': 'NO_FRAME_DATA'
            }), 400
        
        frame_data = data['frame']
        
        # Validate frame data format
        if not frame_data.startswith('data:image/'):
            return jsonify({
                'error': 'Invalid frame data format',
                'code': 'INVALID_FRAME_FORMAT'
            }), 400
        
        # Decode base64 frame
        try:
            frame_bytes = base64.b64decode(frame_data.split(',')[1])
            frame_array = np.frombuffer(frame_bytes, dtype=np.uint8)
            frame = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)
        except Exception as e:
            return jsonify({
                'error': f'Frame decode error: {str(e)}',
                'code': 'FRAME_DECODE_ERROR'
            }), 400
        
        if frame is None:
            return jsonify({
                'error': 'Unable to decode frame data',
                'code': 'FRAME_DECODE_FAILED'
            }), 400
        
        # Validate frame dimensions
        if frame.shape[0] < 50 or frame.shape[1] < 50:
            return jsonify({
                'error': 'Frame too small for analysis',
                'code': 'FRAME_TOO_SMALL'
            }), 400
        
        # Process frame
        start_time = time.time()
        
        # Resize and normalize frame
        frame_resized = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB) / 255.0
        
        # Create tensor with repeated frames
        video_tensor = np.array([frame_rgb] * FRAMES_PER_CLIP)[None, ...]
        
        # Run prediction
        prediction = model.predict(video_tensor, verbose=0)
        probability = float(prediction[0][0])
        
        processing_time = time.time() - start_time
        
        # Determine label and confidence
        threshold = 0.5
        label = "Non-Violence" if probability > threshold else "Violence"
        confidence = probability if label == "Non-Violence" else 1 - probability
        
        result = {
            'label': label,
            'confidence': round(confidence * 100, 2),
            'probability': round(probability * 100, 2),
            'timestamp': datetime.now().isoformat(),
            'frame_info': {
                'original_size': f"{frame.shape[1]}x{frame.shape[0]}",
                'processed_size': f"{IMG_SIZE}x{IMG_SIZE}",
                'channels': frame.shape[2] if len(frame.shape) > 2 else 1
            },
            'performance': {
                'processing_time': round(processing_time * 1000, 1)  # in milliseconds
            }
        }
        
        return jsonify(result)
    
    except Exception as e:
        logger.error(f"❌ Error processing webcam frame: {str(e)}")
        return jsonify({
            'error': f'Frame processing error: {str(e)}',
            'code': 'PROCESSING_ERROR'
        }), 500

@app.route('/status')
def status():
    """Return comprehensive system status"""
    
    # Get system information
    import psutil
    import platform
    
    try:
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
    except:
        cpu_percent = 0
        memory = None
        disk = None
    
    status_data = {
        'model_loaded': model_loaded,
        'model_path': MODEL_PATH,
        'model_load_time': model_load_time,
        'frames_per_clip': FRAMES_PER_CLIP,
        'img_size': IMG_SIZE,
        'server_time': datetime.now().isoformat(),
        'upload_folder': app.config['UPLOAD_FOLDER'],
        'max_content_length': app.config['MAX_CONTENT_LENGTH'],
        'max_content_length_mb': app.config['MAX_CONTENT_LENGTH'] / (1024*1024),
        'allowed_extensions': list(ALLOWED_EXTENSIONS),
        'system_info': {
            'tensorflow_version': tf.__version__,
            'opencv_version': cv2.__version__,
            'python_version': platform.python_version(),
            'platform': platform.system(),
            'cpu_percent': cpu_percent,
            'memory_percent': memory.percent if memory else 0,
            'disk_percent': (disk.used / disk.total * 100) if disk else 0
        },
        'performance': {
            'model_ready': model_loaded,
            'uptime_seconds': time.time() - (model_load_time or time.time()) if model_load_time else 0
        }
    }
    
    return jsonify(status_data)

@app.route('/config', methods=['GET', 'POST'])
def config():
    """Manage configuration settings with validation"""
    config_file = 'config.json'
    
    if request.method == 'POST':
        try:
            data = request.get_json()
            if not data:
                return jsonify({
                    'error': 'No configuration data provided',
                    'code': 'NO_DATA'
                }), 400
            
            # Validate configuration data
            valid_keys = {
                'default_threshold', 'default_frame_skip', 'show_fps', 
                'show_confidence', 'auto_save_logs', 'export_format', 'updated_at'
            }
            
            # Validate specific fields
            if 'default_threshold' in data:
                threshold = data['default_threshold']
                if not isinstance(threshold, (int, float)) or not 0 <= threshold <= 1:
                    return jsonify({
                        'error': 'Invalid threshold value. Must be between 0 and 1.',
                        'code': 'INVALID_THRESHOLD'
                    }), 400
            
            if 'default_frame_skip' in data:
                frame_skip = data['default_frame_skip']
                if not isinstance(frame_skip, int) or not 1 <= frame_skip <= 10:
                    return jsonify({
                        'error': 'Invalid frame skip value. Must be between 1 and 10.',
                        'code': 'INVALID_FRAME_SKIP'
                    }), 400
            
            # Filter only valid keys
            filtered_data = {k: v for k, v in data.items() if k in valid_keys}
            
            # Add metadata
            filtered_data['updated_at'] = datetime.now().isoformat()
            filtered_data['version'] = '2.0'
            filtered_data['updated_by'] = request.remote_addr
            
            # Save configuration
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(filtered_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"⚙️ Configuration updated successfully by {request.remote_addr}")
            return jsonify({
                'message': 'Configuration updated successfully',
                'config': filtered_data
            })
            
        except Exception as e:
            logger.error(f"❌ Error updating config: {str(e)}")
            return jsonify({
                'error': f'Configuration update error: {str(e)}',
                'code': 'UPDATE_ERROR'
            }), 500
    
    else:
        try:
            if os.path.exists(config_file):
                with open(config_file, 'r', encoding='utf-8') as f:
                    config_data = json.load(f)
            else:
                # Default configuration
                config_data = {
                    'default_threshold': 0.5,
                    'default_frame_skip': 1,
                    'show_fps': True,
                    'show_confidence': True,
                    'auto_save_logs': True,
                    'export_format': 'json',
                    'version': '2.0',
                    'created_at': datetime.now().isoformat(),
                    'created_by': request.remote_addr
                }
                
                # Save default config
                with open(config_file, 'w', encoding='utf-8') as f:
                    json.dump(config_data, f, indent=2, ensure_ascii=False)
            
            return jsonify(config_data)
            
        except Exception as e:
            logger.error(f"❌ Error reading config: {str(e)}")
            return jsonify({
                'error': f'Configuration read error: {str(e)}',
                'code': 'READ_ERROR'
            }), 500

@app.route('/health')
def health_check():
    """Comprehensive health check endpoint"""
    health_status = {
        'status': 'healthy' if model_loaded else 'degraded',
        'model_loaded': model_loaded,
        'model_load_time': model_load_time,
        'timestamp': datetime.now().isoformat(),
        'version': '2.0',
        'uptime_seconds': time.time() - (model_load_time or time.time()) if model_load_time else 0,
        'checks': {
            'model': 'pass' if model_loaded else 'fail',
            'upload_folder': 'pass' if os.path.exists(app.config['UPLOAD_FOLDER']) else 'fail',
            'config_writable': 'pass' if os.access('.', os.W_OK) else 'fail'
        }
    }
    
    status_code = 200 if health_status['status'] == 'healthy' else 503
    return jsonify(health_status), status_code

# Error handlers
@app.errorhandler(413)
def too_large(e):
    """Handle file too large error"""
    return jsonify({
        'error': 'File too large. Maximum allowed size is 100MB.',
        'max_size_mb': 100,
        'code': 'FILE_TOO_LARGE'
    }), 413

@app.errorhandler(404)
def not_found(e):
    """Handle 404 errors"""
    return jsonify({
        'error': 'Endpoint not found',
        'code': 'NOT_FOUND'
    }), 404

@app.errorhandler(405)
def method_not_allowed(e):
    """Handle method not allowed errors"""
    return jsonify({
        'error': 'Method not allowed for this endpoint',
        'code': 'METHOD_NOT_ALLOWED'
    }), 405

@app.errorhandler(500)
def internal_error(e):
    """Handle internal server errors"""
    logger.error(f"❌ Internal server error: {str(e)}")
    return jsonify({
        'error': 'Internal server error. Please try again later.',
        'code': 'INTERNAL_ERROR'
    }), 500

# Request logging middleware
@app.before_request
def log_request_info():
    """Log request information"""
    if request.endpoint not in ['static', 'health']:
        logger.info(f"📥 {request.method} {request.path} from {request.remote_addr}")

@app.after_request
def log_response_info(response):
    """Log response information"""
    if request.endpoint not in ['static', 'health']:
        logger.info(f"📤 {response.status_code} for {request.method} {request.path}")
    return response

if __name__ == '__main__':
    logger.info("🚀 Starting Violence Detection AI Web Application...")
    logger.info("=" * 60)
    logger.info(f"📁 Model path: {MODEL_PATH}")
    logger.info(f"🔗 Model source: Google Drive")
    logger.info(f"📂 Upload folder: {app.config['UPLOAD_FOLDER']}")
    logger.info(f"📊 Max file size: {app.config['MAX_CONTENT_LENGTH'] / (1024*1024):.0f}MB")
    logger.info(f"🔧 TensorFlow version: {tf.__version__}")
    logger.info(f"📹 OpenCV version: {cv2.__version__}")
    logger.info(f"🎬 Input parameters: {FRAMES_PER_CLIP} frames, {IMG_SIZE}x{IMG_SIZE}px")
    logger.info("=" * 60)
    
    # Start the application
    app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)
