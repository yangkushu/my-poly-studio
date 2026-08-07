"""MiniMax-H3 视频生成 Tool。"""
import argparse
import base64
import json
import logging
import os
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Literal, Optional
from urllib.parse import urlparse

import requests
from dotenv import load_dotenv
from langchain_core.tools import tool
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent.parent.parent
ENV_PATH = BASE_DIR / ".env"
if ENV_PATH.exists():
    load_dotenv(ENV_PATH)

MINIMAX_API_KEY = os.getenv("MINIMAX_API_KEY", "").strip()
MINIMAX_BASE_URL = os.getenv("MINIMAX_BASE_URL", "https://api.minimaxi.com").rstrip("/")
POLL_INTERVAL_SECONDS = float(os.getenv("VIDEO_POLL_INTERVAL_SECONDS", "5"))
POLL_TIMEOUT_SECONDS = int(os.getenv("VIDEO_POLL_TIMEOUT_SECONDS", "600"))

VIDEOS_DIR = BASE_DIR / "storage" / "videos"
VIDEOS_DIR.mkdir(parents=True, exist_ok=True)

ASPECT_RATIOS = ("adaptive", "21:9", "16:9", "4:3", "1:1", "3:4", "9:16")
MAX_REQUEST_BYTES = 64 * 1024 * 1024
IMAGE_SUFFIXES = {".jpg": "jpeg", ".jpeg": "jpeg", ".png": "png", ".webp": "webp", ".heic": "heic", ".heif": "heif"}
AUDIO_SUFFIXES = {".wav": "wav", ".mp3": "mp3"}


def _local_path_from_storage_url(media_url: str) -> Optional[Path]:
    """将项目的 /storage URL 解析为本地文件；公网 URL、data URI 和 mm_file URI 原样返回。"""
    path = media_url
    if media_url.startswith(("http://", "https://")):
        parsed = urlparse(media_url)
        if parsed.hostname not in {"localhost", "127.0.0.1", "0.0.0.0"}:
            return None
        path = parsed.path
    if not path.startswith("/storage/"):
        return None
    return BASE_DIR / path.lstrip("/")


def _to_minimax_media_url(media_url: str, media_kind: Literal["image", "video", "audio"]) -> tuple[str, int]:
    """将项目本地媒体转为 MiniMax 接受的 data URI，并返回其请求体字节估算。"""
    local_path = _local_path_from_storage_url(media_url)
    if local_path is None:
        return media_url, len(media_url.encode("utf-8")) if media_url.startswith("data:") else 0
    if not local_path.exists():
        raise FileNotFoundError(f"本地媒体不存在: {local_path}")

    suffix = local_path.suffix.lower()
    if media_kind == "image":
        subtype = IMAGE_SUFFIXES.get(suffix)
        if subtype is None:
            raise ValueError("MiniMax H3 本地图片仅支持 jpg、jpeg、png、webp、heic、heif")
        mime_type = f"image/{subtype}"
        max_bytes = 30 * 1024 * 1024
    elif media_kind == "video":
        if suffix != ".mp4":
            raise ValueError("MiniMax H3 本地参考视频仅支持 mp4 data URI；mov 请使用公网 URL 或 mm_file://")
        mime_type = "video/mp4"
        max_bytes = 50 * 1024 * 1024
    else:
        subtype = AUDIO_SUFFIXES.get(suffix)
        if subtype is None:
            raise ValueError("MiniMax H3 本地参考音频仅支持 wav、mp3")
        mime_type = f"audio/{subtype}"
        max_bytes = 15 * 1024 * 1024

    size = local_path.stat().st_size
    if size > max_bytes:
        raise ValueError(f"本地{media_kind}文件超过 MiniMax H3 的 {max_bytes // 1024 // 1024} MB 限制: {local_path}")
    encoded = base64.b64encode(local_path.read_bytes()).decode("ascii")
    data_url = f"data:{mime_type};base64,{encoded}"
    return data_url, len(data_url.encode("utf-8"))


def _media_content(media_url: str, media_kind: Literal["image", "video", "audio"], role: str) -> tuple[dict, int]:
    value, encoded_size = _to_minimax_media_url(media_url, media_kind)
    field_name = f"{media_kind}_url"
    return {"type": field_name, field_name: {"url": value}, "role": role}, encoded_size


def _download_video(video_url: str, prompt: str) -> str:
    response = requests.get(video_url, timeout=300, stream=True)
    response.raise_for_status()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    prompt_part = "".join(char if char.isalnum() or char in "-_" else "_" for char in prompt[:24]).strip("_")
    filename = f"minimax_h3_{timestamp}_{uuid.uuid4().hex[:8]}_{prompt_part}.mp4" if prompt_part else f"minimax_h3_{timestamp}_{uuid.uuid4().hex[:8]}.mp4"
    target = VIDEOS_DIR / filename
    with target.open("wb") as output:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                output.write(chunk)
    return f"/storage/videos/{filename}"


class GenerateMiniMaxH3VideoInput(BaseModel):
    """MiniMax-H3 视频生成工具输入。"""

    prompt: str = Field(description="必填。描述期待的视频画面、动作、镜头和风格；最多 7000 字符。")
    mode: Literal["text", "image", "reference"] = Field(
        default="text",
        description="生成模式：text=纯文生视频；image=首帧或首尾帧图生；reference=使用参考图片、视频或音频生成视频。",
    )
    start_image_url: Optional[str] = Field(default=None, description="image 模式的首帧图片。可为公网 URL、mm_file://{file_id}、data URI 或项目 /storage/... 图片路径。")
    end_image_url: Optional[str] = Field(default=None, description="image 模式可选尾帧图片；必须与 start_image_url 一起传入。")
    reference_image_urls: list[str] = Field(default_factory=list, description="reference 模式可选的 1–9 张参考图；不可与首帧或尾帧混用。")
    reference_video_urls: list[str] = Field(default_factory=list, description="reference 模式可选的 1–3 个参考视频；不可与首帧或尾帧混用。")
    reference_audio_urls: list[str] = Field(default_factory=list, description="reference 模式可选的 1–3 个参考音频；不可与首帧或尾帧混用。")
    duration: Literal[4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15] = Field(default=5, description="输出时长（秒），仅支持 4–15 的整数。")
    resolution: Literal["768P", "2K"] = Field(default="768P", description="输出分辨率，768P 或 2K。")
    aspect_ratio: Literal["adaptive", "21:9", "16:9", "4:3", "1:1", "3:4", "9:16"] = Field(default="16:9", description="text 模式必须指定具体比例；image 模式由图片决定并固定 adaptive；reference 模式可指定或使用 adaptive。")
    aigc_watermark: bool = Field(default=False, description="是否添加 AIGC 标识水印，默认不添加。")


@tool("generate_minimax_h3_video", args_schema=GenerateMiniMaxH3VideoInput)
def generate_minimax_h3_video_tool(
    prompt: str,
    mode: Literal["text", "image", "reference"] = "text",
    start_image_url: Optional[str] = None,
    end_image_url: Optional[str] = None,
    reference_image_urls: Optional[list[str]] = None,
    reference_video_urls: Optional[list[str]] = None,
    reference_audio_urls: Optional[list[str]] = None,
    duration: Literal[4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15] = 5,
    resolution: Literal["768P", "2K"] = "768P",
    aspect_ratio: Literal["adaptive", "21:9", "16:9", "4:3", "1:1", "3:4", "9:16"] = "16:9",
    aigc_watermark: bool = False,
) -> str:
    """使用 MiniMax-H3 生成视频，并在任务完成后下载视频到本项目 storage/videos。"""
    if not MINIMAX_API_KEY:
        logger.error("MiniMax H3 视频生成未开始: 未配置 MINIMAX_API_KEY")
        return json.dumps({"error": "未配置 MINIMAX_API_KEY（请在 backend/.env 设置，可参考 env.example）"}, ensure_ascii=False)

    reference_image_urls = reference_image_urls or []
    reference_video_urls = reference_video_urls or []
    reference_audio_urls = reference_audio_urls or []
    reference_count = len(reference_image_urls) + len(reference_video_urls) + len(reference_audio_urls)

    if not prompt.strip():
        return json.dumps({"error": "prompt 不能为空"}, ensure_ascii=False)
    if len(prompt) > 7000:
        return json.dumps({"error": "prompt 最多 7000 个字符"}, ensure_ascii=False)
    if len(reference_image_urls) > 9 or len(reference_video_urls) > 3 or len(reference_audio_urls) > 3:
        return json.dumps({"error": "reference_image_urls 最多 9 项，reference_video_urls 和 reference_audio_urls 各最多 3 项"}, ensure_ascii=False)
    if mode == "text" and (start_image_url or end_image_url or reference_count):
        return json.dumps({"error": "text 模式不能传入图片、视频或音频参考素材"}, ensure_ascii=False)
    if mode == "image" and not start_image_url:
        return json.dumps({"error": "image 模式必须提供 start_image_url"}, ensure_ascii=False)
    if mode == "image" and reference_count:
        return json.dumps({"error": "image 模式不能与参考图片、视频或音频混用"}, ensure_ascii=False)
    if mode == "reference" and (start_image_url or end_image_url):
        return json.dumps({"error": "reference 模式不能与首帧或尾帧图片混用"}, ensure_ascii=False)
    if mode == "reference" and not reference_count:
        return json.dumps({"error": "reference 模式至少需要一项参考图片、视频或音频"}, ensure_ascii=False)
    if mode == "text" and aspect_ratio == "adaptive":
        return json.dumps({"error": "text 模式的 aspect_ratio 不能为 adaptive"}, ensure_ascii=False)

    try:
        content = [{"type": "text", "text": prompt}]
        encoded_bytes = len(json.dumps(content, ensure_ascii=False).encode("utf-8"))
        if mode == "image":
            first_frame, size = _media_content(start_image_url, "image", "first_frame")
            content.append(first_frame)
            encoded_bytes += size
            if end_image_url:
                last_frame, size = _media_content(end_image_url, "image", "last_frame")
                content.append(last_frame)
                encoded_bytes += size
            request_ratio = "adaptive"
        elif mode == "reference":
            request_ratio = aspect_ratio
            for media_url in reference_image_urls:
                item, size = _media_content(media_url, "image", "reference_image")
                content.append(item)
                encoded_bytes += size
            for media_url in reference_video_urls:
                item, size = _media_content(media_url, "video", "reference_video")
                content.append(item)
                encoded_bytes += size
            for media_url in reference_audio_urls:
                item, size = _media_content(media_url, "audio", "reference_audio")
                content.append(item)
                encoded_bytes += size
        else:
            request_ratio = aspect_ratio
        if encoded_bytes > MAX_REQUEST_BYTES:
            raise ValueError("本地媒体编码后的请求体超过 MiniMax H3 的 64 MB 限制；请改用公网 URL 或 mm_file://")

        payload = {
            "model": "MiniMax-H3",
            "content": content,
            "resolution": resolution,
            "duration": duration,
            "ratio": request_ratio,
            "aigc_watermark": aigc_watermark,
        }
        headers = {"Authorization": f"Bearer {MINIMAX_API_KEY}", "Content-Type": "application/json"}
        logger.info("提交 MiniMax H3 视频任务: mode=%s duration=%ss resolution=%s ratio=%s", mode, duration, resolution, request_ratio)
        submit = requests.post(f"{MINIMAX_BASE_URL}/v2/video_generation", json=payload, headers=headers, timeout=60)
        if not submit.ok:
            raise RuntimeError(f"提交任务失败: status={submit.status_code}, body={submit.text}")
        task_id = submit.json().get("task_id")
        if not task_id:
            raise RuntimeError(f"提交响应未包含 task_id: {submit.text}")

        deadline = time.monotonic() + POLL_TIMEOUT_SECONDS
        last_status: Optional[str] = None
        while time.monotonic() < deadline:
            query = requests.get(f"{MINIMAX_BASE_URL}/v2/query/video_generation/{task_id}", headers=headers, timeout=60)
            if not query.ok:
                raise RuntimeError(f"查询任务失败: status={query.status_code}, body={query.text}")
            task = query.json().get("task")
            if not isinstance(task, dict):
                raise RuntimeError(f"查询响应未包含 task: {query.text}")
            status = str(task.get("status", "")).lower()
            if status != last_status:
                logger.info("MiniMax H3 视频任务状态变化: task_id=%s status=%s", task_id, status or "unknown")
                last_status = status
            if status in {"failed", "cancelled"}:
                raise RuntimeError(f"任务{status}: {json.dumps(task, ensure_ascii=False)}")
            if status == "succeeded":
                result_content = task.get("content")
                video_url = result_content.get("url") if isinstance(result_content, dict) else None
                if not video_url:
                    raise RuntimeError(f"成功任务未包含视频 URL: {json.dumps(task, ensure_ascii=False)}")
                local_path = _download_video(video_url, prompt)
                return json.dumps({
                    "video_url": local_path,
                    "original_url": video_url,
                    "local_path": local_path,
                    "task_id": task_id,
                    "provider": "minimax",
                    "model": "MiniMax-H3",
                    "mode": mode,
                    "prompt": prompt,
                    "resolution": task.get("resolution", resolution),
                    "duration": task.get("duration", duration),
                    "ratio": task.get("ratio", request_ratio),
                    "usage": task.get("usage"),
                    "message": "视频已生成并保存到本地",
                }, ensure_ascii=False)
            time.sleep(POLL_INTERVAL_SECONDS)
        raise TimeoutError(f"任务超时: {POLL_TIMEOUT_SECONDS} 秒内未完成（task_id={task_id}）")
    except Exception as error:
        logger.exception("MiniMax H3 视频生成失败: mode=%s", mode)
        return json.dumps({"error": f"生成视频时出错: {error}"}, ensure_ascii=False)


def main() -> None:
    """命令行测试入口；使用有效 API Key 会产生 MiniMax 视频生成费用。"""
    parser = argparse.ArgumentParser(description="测试 MiniMax-H3 视频生成 Tool")
    parser.add_argument("--prompt", default="一只小狗在阳光下的草地上奔跑", help="视频提示词")
    parser.add_argument("--mode", choices=("text", "image", "reference"), default="text")
    parser.add_argument("--start-image-url")
    parser.add_argument("--end-image-url")
    parser.add_argument("--reference-image-url", dest="reference_image_urls", action="append", default=[])
    parser.add_argument("--reference-video-url", dest="reference_video_urls", action="append", default=[])
    parser.add_argument("--reference-audio-url", dest="reference_audio_urls", action="append", default=[])
    parser.add_argument("--duration", type=int, choices=range(4, 16), default=5)
    parser.add_argument("--resolution", choices=("768P", "2K"), default="768P")
    parser.add_argument("--aspect-ratio", choices=ASPECT_RATIOS, default="16:9")
    parser.add_argument("--aigc-watermark", action="store_true")
    args = parser.parse_args()

    if args.mode == "image" and not args.start_image_url:
        parser.error("--mode image 时必须提供 --start-image-url")
    if args.mode == "reference" and not (args.reference_image_urls or args.reference_video_urls or args.reference_audio_urls):
        parser.error("--mode reference 时至少提供一项参考媒体")
    logging.basicConfig(level=logging.INFO)
    print(generate_minimax_h3_video_tool.invoke(vars(args)))


if __name__ == "__main__":
    main()
