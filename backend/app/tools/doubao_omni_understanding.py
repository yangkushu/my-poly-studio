"""火山方舟 Doubao-Seed-2.0 Lite 全模态理解工具。

使用 ``doubao-seed-2-0-lite-260428`` 对图片、音频和视频进行原生理解。
该版本可直接分析语音情绪、环境声和音乐，并可将这些声音线索与视频画面联合推理。
"""

import json
import logging
import mimetypes
import os
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

from dotenv import load_dotenv
from langchain_core.tools import tool
from pydantic import BaseModel, Field
import requests

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent.parent.parent
ENV_PATH = BASE_DIR / ".env"
if ENV_PATH.exists():
    load_dotenv(ENV_PATH)

VOLCANO_API_KEY = os.getenv("VOLCANO_API_KEY", "").strip()
VOLCANO_BASE_URL = os.getenv(
    "VOLCANO_BASE_URL", "https://ark.cn-beijing.volces.com/api/v3"
).strip()
# 260428 是支持原生音频输入的全模态版本。也可以设置为方舟控制台创建的 ep-* 接入点。
DOUBAO_OMNI_MODEL = os.getenv(
    "DOUBAO_OMNI_MODEL", "doubao-seed-2-0-lite-260428"
).strip()
IMAGE_MIME = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
    ".gif": "image/gif",
    ".bmp": "image/bmp",
}
AUDIO_MIME = {
    ".mp3": "audio/mpeg",
    ".wav": "audio/wav",
    ".m4a": "audio/mp4",
    ".aac": "audio/aac",
}
VIDEO_MIME = {
    ".mp4": "video/mp4",
    ".mov": "video/quicktime",
    ".avi": "video/x-msvideo",
}


def _is_remote_url(value: str) -> bool:
    return urlparse(value).scheme in {"http", "https"}


def _resolve_local_path(media_path: str) -> Path:
    """解析本地路径，并限制 /storage 相对路径位于 backend 目录。"""
    file_path = BASE_DIR / media_path.lstrip("/") if media_path.startswith("/storage/") else Path(media_path)
    if not file_path.exists() or not file_path.is_file():
        raise FileNotFoundError(f"文件不存在: {file_path}")
    return file_path


def _extension(media_path: str) -> str:
    return Path(urlparse(media_path).path).suffix.lower()


def _detect_media_type(media_path: str) -> str:
    ext = _extension(media_path)
    if ext in IMAGE_MIME:
        return "image"
    if ext in AUDIO_MIME:
        return "audio"
    if ext in VIDEO_MIME:
        return "video"
    raise ValueError(
        f"无法识别媒体格式 '{ext}'，支持图片 {list(IMAGE_MIME)}、"
        f"音频 {list(AUDIO_MIME)}、视频 {list(VIDEO_MIME)}"
    )


def _upload_file_to_ark(file_path: Path) -> str:
    """上传本地媒体到方舟 Files API，并返回可用于 Responses 的 file_id。"""
    mime_type = mimetypes.guess_type(file_path.name)[0] or "application/octet-stream"
    with file_path.open("rb") as media_file:
        response = requests.post(
            f"{VOLCANO_BASE_URL.rstrip('/')}/files",
            headers={"Authorization": f"Bearer {VOLCANO_API_KEY}"},
            data={"purpose": "user_data"},
            files={"file": (file_path.name, media_file, mime_type)},
            timeout=120,
        )
    try:
        response.raise_for_status()
    except requests.HTTPError as error:
        raise RuntimeError(f"方舟文件上传失败: {response.text}") from error

    file_id = response.json().get("id")
    if not file_id:
        raise RuntimeError(f"方舟文件上传未返回 id: {response.json()}")
    return file_id


def _delete_ark_file(file_id: str) -> None:
    """尽力清理本次调用上传的方舟临时文件；清理失败不影响理解结果。"""
    try:
        response = requests.delete(
            f"{VOLCANO_BASE_URL.rstrip('/')}/files/{file_id}",
            headers={"Authorization": f"Bearer {VOLCANO_API_KEY}"},
            timeout=15,
        )
        response.raise_for_status()
    except requests.RequestException as error:
        logger.warning("方舟临时文件删除失败: file_id=%s, error=%s", file_id, error)


def _build_media_content(media_path: str, media_type: str) -> tuple[dict, Optional[str]]:
    """构建方舟 Responses API 的媒体 content，并返回需清理的本地上传 file_id。"""
    content_key = {"image": "image_url", "audio": "audio_url", "video": "video_url"}[media_type]
    content_type = {"image": "input_image", "audio": "input_audio", "video": "input_video"}[media_type]
    if _is_remote_url(media_path):
        return {"type": content_type, content_key: media_path}, None

    file_id = _upload_file_to_ark(_resolve_local_path(media_path))
    return {"type": content_type, "file_id": file_id}, file_id


def _call_doubao(input_items: list[dict]) -> str:
    """调用方舟 Responses API，并提取所有 output_text 内容。"""
    response = requests.post(
        f"{VOLCANO_BASE_URL.rstrip('/')}/responses",
        headers={
            "Authorization": f"Bearer {VOLCANO_API_KEY}",
            "Content-Type": "application/json",
        },
        json={"model": DOUBAO_OMNI_MODEL, "input": input_items, "store": False},
        timeout=300,
    )
    try:
        response.raise_for_status()
    except requests.HTTPError as error:
        try:
            detail = response.json()
        except ValueError:
            detail = response.text
        raise RuntimeError(f"方舟 Responses API 调用失败: {detail}") from error

    payload = response.json()
    text_parts = [
        content["text"]
        for output in payload.get("output", [])
        if output.get("type") == "message"
        for content in output.get("content", [])
        if content.get("type") == "output_text" and content.get("text")
    ]
    if not text_parts:
        raise RuntimeError(f"方舟未返回 output_text: {payload}")
    return "".join(text_parts)


class DoubaoOmniUnderstandInput(BaseModel):
    """Doubao-Seed-2.0 Lite 全模态理解输入。"""

    media_path: str = Field(
        description="媒体路径。可传公网 HTTPS URL；本地 /storage/... 文件会自动上传至方舟临时文件服务。"
    )
    question: str = Field(
        description="对媒体的分析问题或指令，例如：分析口播情绪、背景音乐与画面节奏的关系。"
    )
    media_type: Optional[str] = Field(
        default=None,
        description="媒体类型：image、audio 或 video。留空时按文件扩展名识别。",
    )


@tool("doubao_omni_understand", args_schema=DoubaoOmniUnderstandInput)
def doubao_omni_understand_tool(
    media_path: str,
    question: str,
    media_type: Optional[str] = None,
) -> str:
    """使用 Doubao-Seed-2.0 Lite 原生理解图片、音频或视频，并返回文字分析结果。

    适用于声音事件、说话情绪、背景音乐、音画一致性、视频时间线与跨模态问答。
    不执行“仅转写再分析”的两阶段流程。
    """
    try:
        if not VOLCANO_API_KEY:
            return "Error: 未配置 VOLCANO_API_KEY（请在 backend/.env 设置）"

        detected_type = media_type or _detect_media_type(media_path)
        if detected_type not in {"image", "audio", "video"}:
            raise ValueError("media_type 仅支持 image、audio 或 video")

        media_content, uploaded_file_id = _build_media_content(media_path, detected_type)
        input_items = [
            {
                "role": "user",
                "content": [
                    media_content,
                    {"type": "input_text", "text": question},
                ],
            }
        ]
        logger.info(
            "🔍 Doubao 全模态分析: type=%s, media=%s, question=%s...",
            detected_type,
            Path(urlparse(media_path).path).name,
            question[:60],
        )
        try:
            text_response = _call_doubao(input_items)
        finally:
            if uploaded_file_id:
                _delete_ark_file(uploaded_file_id)
        result = {
            "text_response": text_response,
            "media_path": media_path,
            "media_type": detected_type,
            "question": question,
            "model": DOUBAO_OMNI_MODEL,
            "provider": "volcano-doubao",
            "message": "Doubao 全模态理解完成",
        }
        logger.info("✅ Doubao 全模态理解成功: text_len=%s", len(text_response))
        return json.dumps(result, ensure_ascii=False)
    except Exception as error:
        logger.exception("❌ Doubao 全模态理解失败")
        return f"Error: {error}"
