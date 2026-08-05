"""智谱 Vidu 2 视频生成 Tool。"""
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

ZHIPU_API_KEY = os.getenv("ZHIPU_API_KEY", "").strip()
ZHIPU_BASE_URL = os.getenv("ZHIPU_BASE_URL", "https://open.bigmodel.cn/api").rstrip("/")
POLL_INTERVAL_SECONDS = float(os.getenv("VIDEO_POLL_INTERVAL_SECONDS", "5"))
POLL_TIMEOUT_SECONDS = int(os.getenv("VIDEO_POLL_TIMEOUT_SECONDS", "600"))

VIDEOS_DIR = BASE_DIR / "storage" / "videos"
VIDEOS_DIR.mkdir(parents=True, exist_ok=True)

VIDU_MODELS = {
    "image": "vidu2-image",
    "start_end": "vidu2-start-end",
    "reference": "vidu2-reference",
}


def _local_path_from_storage_url(image_url: str) -> Optional[Path]:
    path = image_url
    if image_url.startswith(("http://", "https://")):
        parsed = urlparse(image_url)
        if parsed.hostname not in {"localhost", "127.0.0.1", "0.0.0.0"}:
            return None
        path = parsed.path
    if not path.startswith("/storage/"):
        return None
    return BASE_DIR / path.lstrip("/")


def _zhipu_image(image_url: str) -> str:
    """智谱接受公网 URL 或 data:image/...;base64 格式，本地图片需转换为后者。"""
    local_path = _local_path_from_storage_url(image_url)
    if local_path is None:
        return image_url
    if not local_path.exists():
        raise FileNotFoundError(f"本地图片不存在: {local_path}")
    suffix = local_path.suffix.lower()
    media_type = {".jpg": "jpeg", ".jpeg": "jpeg", ".png": "png", ".webp": "webp"}.get(suffix)
    if media_type is None:
        raise ValueError("Vidu 2 本地图片仅支持 jpg、jpeg、png、webp")
    encoded = base64.b64encode(local_path.read_bytes()).decode("ascii")
    return f"data:image/{media_type};base64,{encoded}"


def _download_video(video_url: str, prompt: str) -> str:
    response = requests.get(video_url, timeout=300, stream=True)
    response.raise_for_status()
    suffix = Path(urlparse(video_url).path).suffix.lower()
    if suffix not in {".mp4", ".mov", ".webm"}:
        suffix = ".mp4"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    prompt_part = "".join(char if char.isalnum() or char in "-_" else "_" for char in prompt[:24]).strip("_")
    filename = f"vidu2_{timestamp}_{uuid.uuid4().hex[:8]}_{prompt_part}{suffix}" if prompt_part else f"vidu2_{timestamp}_{uuid.uuid4().hex[:8]}{suffix}"
    target = VIDEOS_DIR / filename
    with target.open("wb") as output:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                output.write(chunk)
    return f"/storage/videos/{filename}"


class GenerateZhipuVidu2VideoInput(BaseModel):
    """智谱 Vidu 2 统一 Tool 的输入参数。"""

    prompt: str = Field(description="视频内容描述，最多 512 个字符。")
    mode: Literal["image", "start_end", "reference"] = Field(description="图生、首尾帧或多参考图模式。")
    image_urls: list[str] = Field(description="image 恰好 1 张；start_end 恰好 2 张；reference 为 1 至 3 张。")
    movement_amplitude: Literal["auto", "small", "medium", "large"] = Field(default="auto", description="视频运动幅度。")
    with_audio: bool = Field(default=False, description="是否添加背景音乐；Vidu 2 固定生成 4 秒视频。")
    aspect_ratio: Literal["16:9", "9:16", "1:1"] = Field(default="16:9", description="仅 reference 模式有效。")
    size: Optional[Literal["1280x720", "480x360"]] = Field(default=None, description="仅 start_end 可选 1280x720 或 480x360；其他模式固定 1280x720。")


@tool("generate_zhipu_vidu2_video", args_schema=GenerateZhipuVidu2VideoInput)
def generate_zhipu_vidu2_video_tool(
    prompt: str,
    mode: Literal["image", "start_end", "reference"],
    image_urls: list[str],
    movement_amplitude: Literal["auto", "small", "medium", "large"] = "auto",
    with_audio: bool = False,
    aspect_ratio: Literal["16:9", "9:16", "1:1"] = "16:9",
    size: Optional[Literal["1280x720", "480x360"]] = None,
) -> str:
    """使用智谱的 Vidu 2 生成视频，并将三个 Vidu 2 模型合并为一个 Tool。

    image 对应 vidu2-image（1 张首帧）；start_end 对应 vidu2-start-end（2 张首尾帧）；
    reference 对应 vidu2-reference（1–3 张参考图）。三种模式均固定生成 4 秒、720P 视频。
    价格（2026-08-05）：1.25 元/次 ÷ 4 秒 = 0.3125 元/秒。

    返回下载到本地 /storage/videos/ 的 video_url JSON。
    """
    if not ZHIPU_API_KEY:
        return json.dumps({"error": "未配置 ZHIPU_API_KEY（请在 backend/.env 设置，可参考 env.example）"}, ensure_ascii=False)

    expected_counts = {"image": {1}, "start_end": {2}, "reference": {1, 2, 3}}
    if len(image_urls) not in expected_counts[mode]:
        expected = {"image": "恰好 1 张", "start_end": "恰好 2 张", "reference": "1 至 3 张"}[mode]
        return json.dumps({"error": f"{mode} 模式需要 {expected}图片，当前收到 {len(image_urls)} 张"}, ensure_ascii=False)
    if size == "480x360" and mode != "start_end":
        return json.dumps({"error": "480x360 仅支持 start_end 模式"}, ensure_ascii=False)

    try:
        images = [_zhipu_image(image_url) for image_url in image_urls]
        model = VIDU_MODELS[mode]
        payload = {
            "model": model,
            "prompt": prompt,
            "image_url": images[0] if mode == "image" else images,
            "duration": 4,
            "size": size or "1280x720",
            "movement_amplitude": movement_amplitude,
            "with_audio": with_audio,
        }
        if mode == "reference":
            payload["aspect_ratio"] = aspect_ratio

        headers = {"Authorization": f"Bearer {ZHIPU_API_KEY}", "Content-Type": "application/json"}
        submit = requests.post(f"{ZHIPU_BASE_URL}/paas/v4/videos/generations", json=payload, headers=headers, timeout=60)
        if not submit.ok:
            raise RuntimeError(f"提交任务失败: status={submit.status_code}, body={submit.text}")
        task = submit.json()
        task_id = task.get("id")
        if not task_id:
            raise RuntimeError(f"提交响应未包含 id: {json.dumps(task, ensure_ascii=False)}")

        deadline = time.monotonic() + POLL_TIMEOUT_SECONDS
        while time.monotonic() < deadline:
            query = requests.get(f"{ZHIPU_BASE_URL}/paas/v4/async-result/{task_id}", headers=headers, timeout=60)
            if not query.ok:
                raise RuntimeError(f"查询任务失败: status={query.status_code}, body={query.text}")
            result = query.json()
            if result.get("error"):
                error = result["error"]
                raise RuntimeError(error.get("message", str(error)) if isinstance(error, dict) else str(error))
            status = str(result.get("task_status", "")).upper()
            if status in {"FAIL", "FAILED", "ERROR"}:
                raise RuntimeError(f"任务失败: {json.dumps(result, ensure_ascii=False)}")
            video_results = result.get("video_result")
            if isinstance(video_results, list) and video_results and video_results[0].get("url"):
                video_url = video_results[0]["url"]
                local_path = _download_video(video_url, prompt)
                return json.dumps({
                    "video_url": local_path,
                    "original_url": video_url,
                    "cover_image_url": video_results[0].get("cover_image_url"),
                    "local_path": local_path,
                    "task_id": task_id,
                    "provider": "zhipu",
                    "model": model,
                    "mode": mode,
                    "prompt": prompt,
                    "price_yuan_per_request": 1.25,
                    "price_yuan_per_second": 0.3125,
                    "message": "视频已生成并保存到本地",
                }, ensure_ascii=False)
            time.sleep(POLL_INTERVAL_SECONDS)
        raise TimeoutError(f"任务超时: {POLL_TIMEOUT_SECONDS} 秒内未完成（task_id={task_id}）")
    except Exception as error:
        logger.exception("智谱 Vidu 2 生成失败")
        return json.dumps({"error": f"生成视频时出错: {error}"}, ensure_ascii=False)


def main() -> None:
    """命令行测试入口；使用有效 API Key 会产生智谱 Vidu 2 视频生成费用。"""
    parser = argparse.ArgumentParser(description="测试智谱 Vidu 2 视频生成 Tool")
    parser.add_argument("--prompt", default="让画面中的主体自然地向前移动", help="视频提示词")
    parser.add_argument("--mode", choices=("image", "start_end", "reference"), default="image", help="Vidu 2 生成模式")
    parser.add_argument("--image-url", dest="image_urls", action="append", required=True, help="图片公网 URL 或 /storage/images/... 本地路径；可重复传入")
    parser.add_argument("--movement-amplitude", choices=("auto", "small", "medium", "large"), default="auto", help="运动幅度")
    parser.add_argument("--with-audio", action="store_true", help="生成背景音乐")
    parser.add_argument("--aspect-ratio", choices=("16:9", "9:16", "1:1"), default="16:9", help="仅 reference 模式有效")
    parser.add_argument("--size", choices=("1280x720", "480x360"), help="仅 start_end 支持 480x360")
    args = parser.parse_args()

    expected_counts = {"image": {1}, "start_end": {2}, "reference": {1, 2, 3}}
    if len(args.image_urls) not in expected_counts[args.mode]:
        parser.error(f"--mode {args.mode} 收到 {len(args.image_urls)} 张图片，数量不符合要求")
    if args.size == "480x360" and args.mode != "start_end":
        parser.error("--size 480x360 仅支持 --mode start_end")

    logging.basicConfig(level=logging.INFO)
    result = generate_zhipu_vidu2_video_tool.invoke({
        "prompt": args.prompt,
        "mode": args.mode,
        "image_urls": args.image_urls,
        "movement_amplitude": args.movement_amplitude,
        "with_audio": args.with_audio,
        "aspect_ratio": args.aspect_ratio,
        "size": args.size,
    })
    print(result)


if __name__ == "__main__":
    main()
