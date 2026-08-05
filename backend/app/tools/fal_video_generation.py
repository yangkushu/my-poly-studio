"""fal.ai 视频 Tool 共享的 SDK 调用、文件上传与结果下载逻辑。

官方 SDK 文档：https://fal.ai/docs/api-reference/client-libraries/python/fal_client
异步队列文档：https://fal.ai/docs/documentation/model-apis/inference/queue
"""
import json
import logging
import base64
import mimetypes
import os
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlparse

import requests
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent.parent.parent
ENV_PATH = BASE_DIR / ".env"
if ENV_PATH.exists():
    load_dotenv(ENV_PATH)

FAL_API_KEY = os.getenv("FAL_KEY", "").strip()
FAL_USD_CNY_RATE = float(os.getenv("FAL_USD_CNY_RATE", "6.77"))
POLL_INTERVAL_SECONDS = float(os.getenv("VIDEO_POLL_INTERVAL_SECONDS", "5"))
POLL_TIMEOUT_SECONDS = int(os.getenv("VIDEO_POLL_TIMEOUT_SECONDS", "600"))

VIDEOS_DIR = BASE_DIR / "storage" / "videos"
VIDEOS_DIR.mkdir(parents=True, exist_ok=True)


def _import_fal_client() -> Any:
    """延迟导入，避免未安装 SDK 时阻断其他 Tool 的加载。"""
    try:
        import fal_client
    except ImportError as error:
        raise RuntimeError("未安装 fal-client；请在 backend 环境中执行 pip install -r requirements.txt") from error
    return fal_client


def create_fal_client() -> Any:
    """创建使用当前 FAL_KEY 的 SDK 客户端。"""
    if not FAL_API_KEY:
        raise RuntimeError("未配置 FAL_KEY（请在 backend/.env 设置，可参考 env.example）")
    return _import_fal_client().SyncClient(key=FAL_API_KEY, default_timeout=60)


def _local_path_from_storage_url(file_url: str) -> Optional[Path]:
    """将 /storage/... 或本机 storage URL 映射到后端文件路径。"""
    path = file_url
    if file_url.startswith(("http://", "https://")):
        parsed = urlparse(file_url)
        if parsed.hostname not in {"localhost", "127.0.0.1", "0.0.0.0"}:
            return None
        path = parsed.path
    if not path.startswith("/storage/"):
        return None
    return BASE_DIR / path.lstrip("/")


def resolve_input_url(file_url: str, client: Any) -> str:
    """公网 URL/Data URI 直传；本地存储文件按 data URI 传给 fal。"""
    if file_url.startswith("data:"):
        return file_url
    local_path = _local_path_from_storage_url(file_url)
    if local_path is None:
        return file_url
    if not local_path.exists():
        raise FileNotFoundError(f"本地文件不存在: {local_path}")

    # 与火山视频工具保持一致：本地文件直接转成 data URI，不经过 fal storage
    # upload。这样既不会受中文文件名的 ASCII 编码影响，也不会触发账号不支持
    # 的 storage_type=gcs。fal 的文件输入支持 Base64 data URI。
    content_type = mimetypes.guess_type(str(local_path))[0] or "image/jpeg"
    with local_path.open("rb") as source:
        encoded = base64.b64encode(source.read()).decode("ascii")
    logger.info("📁 本地图片已转为 data URI: %s, content_type=%s", local_path, content_type)
    return f"data:{content_type};base64,{encoded}"


def download_video(video_url: str, provider: str, prompt: str) -> str:
    """下载 fal 返回的临时视频 URL 到项目 storage/videos。"""
    response = requests.get(video_url, timeout=300, stream=True)
    response.raise_for_status()
    suffix = Path(urlparse(video_url).path).suffix.lower()
    if suffix not in {".mp4", ".mov", ".webm"}:
        suffix = ".mp4"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    prompt_part = "".join(char if char.isalnum() or char in "-_" else "_" for char in prompt[:24]).strip("_")
    filename_base = f"fal_{provider}_{timestamp}_{uuid.uuid4().hex[:8]}"
    filename = f"{filename_base}_{prompt_part}{suffix}" if prompt_part else f"{filename_base}{suffix}"
    target = VIDEOS_DIR / filename
    with target.open("wb") as output:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                output.write(chunk)
    return f"/storage/videos/{filename}"


def generate_fal_video(endpoint: str, payload: dict[str, Any], prompt: str, provider: str) -> dict[str, Any]:
    """提交 fal 队列任务、轮询结果并下载视频。"""
    fal_client = _import_fal_client()
    client = create_fal_client()
    handle = client.submit(endpoint, payload)
    deadline = time.monotonic() + POLL_TIMEOUT_SECONDS

    while time.monotonic() < deadline:
        status = handle.status()
        if isinstance(status, fal_client.Completed):
            result = handle.get()
            video = result.get("video") if isinstance(result, dict) else None
            video_url = video.get("url") if isinstance(video, dict) else None
            if not video_url:
                raise RuntimeError(f"fal 任务完成但响应未包含 video.url: {json.dumps(result, ensure_ascii=False)}")
            local_path = download_video(video_url, provider, prompt)
            return {
                "task_id": handle.request_id,
                "original_url": video_url,
                "video_url": local_path,
                "local_path": local_path,
                "fal_result": result,
            }
        time.sleep(POLL_INTERVAL_SECONDS)

    try:
        handle.cancel()
    except Exception:  # 超时后的取消失败不掩盖原始超时错误
        logger.warning("取消超时的 fal 任务失败: %s", handle.request_id, exc_info=True)
    raise TimeoutError(f"fal 任务超时: {POLL_TIMEOUT_SECONDS} 秒内未完成（task_id={handle.request_id}）")
