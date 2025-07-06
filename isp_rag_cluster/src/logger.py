
import logging
import sys
import codecs
from pathlib import Path

from datetime import datetime

class SafeStreamHandler(logging.StreamHandler):
    def __init__(self, stream=None):
        super().__init__(stream)
        self.encoding = 'utf-8'
        if sys.platform == 'win32':
            if stream is None:
                self.stream = codecs.getwriter('utf-8')(sys.stdout.buffer, errors='replace')
            elif hasattr(stream, 'buffer'):
                self.stream = codecs.getwriter('utf-8')(stream.buffer, errors='replace')

    def format(self, record):
        """Format the specified record safely."""
        try:
            message = super().format(record)
            if sys.platform == 'win32':
                # Windows에서는 ASCII로 변환
                return message.encode('ascii', errors='replace').decode('ascii')
            return message
        except Exception as e:
            return f"Error formatting log message: {str(e)}"

    def emit(self, record):
        """Emit a record safely."""
        try:
            msg = self.format(record)
            stream = self.stream
            
            try:
                if isinstance(msg, str):
                    stream.write(msg + self.terminator)
                else:
                    stream.write(str(msg) + self.terminator)
                self.flush()
            except (UnicodeEncodeError, UnicodeDecodeError):
                # 인코딩 에러 발생 시 ASCII로 변환 시도
                try:
                    safe_msg = str(msg).encode('ascii', errors='replace').decode('ascii')
                    stream.write(safe_msg + self.terminator)
                    self.flush()
                except Exception as e:
                    sys.stderr.write(f"Severe logging error: {str(e)}\n")
                    self.handleError(record)
            except Exception as e:
                sys.stderr.write(f"Logging error: {str(e)}\n")
                self.handleError(record)
        except Exception:
            self.handleError(record)

def setup_logging(cfg, output_dir: Path):
    """Logging setup"""
    # Create log directory
    log_dir = output_dir / cfg.general.logging.log_path
    log_dir.mkdir(parents=True, exist_ok=True)

    # Log file path
    log_file = log_dir / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    # Logger setup
    logger = logging.getLogger('emotion_analysis')
    logger.setLevel(logging.DEBUG)

    # Remove existing handlers
    logger.handlers.clear()

    # Windows environment setup for UTF-8 output
    if sys.platform == 'win32':
        import locale
        locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')

    try:
        # File handler setup
        file_handler = logging.FileHandler(log_file, encoding='utf-8', mode='w')
        file_handler.setLevel(logging.DEBUG)

        # Console handler setup
        console_handler = SafeStreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        
        # Formatter setup
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    except Exception as e:
        print(f"Error setting up logging: {e}")
        # Default console logger setup
        console_handler = SafeStreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger

def safe_log(logger, level: str, message: str):
    """Safe logging function"""
    try:
        # Convert non-string types to string
        if not isinstance(message, str):
            message = str(message)
        
        # Log message based on level
        log_func = getattr(logger, level)
        log_func(message)
            
    except Exception as e:
        # Log failure fallback to default output
        print(f"Logging failed: {str(e)}")
        print(f"Original message: {message}")
