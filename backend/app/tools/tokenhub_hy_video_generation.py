"""腾讯云 TokenHub HY-Video-1.5 视频生成 Tool。"""
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

TOKENHUB_API_KEY = os.getenv("TOKENHUB_API_KEY", "").strip()
TOKENHUB_BASE_URL = os.getenv("TOKENHUB_BASE_URL", "https://tokenhub.tencentmaas.com").rstrip("/")
TOKENHUB_VIDEO_MODEL = "hy-video-1.5"
POLL_INTERVAL_SECONDS = float(os.getenv("VIDEO_POLL_INTERVAL_SECONDS", "5"))
POLL_TIMEOUT_SECONDS = int(os.getenv("VIDEO_POLL_TIMEOUT_SECONDS", "600"))

VIDEOS_DIR = BASE_DIR / "storage" / "videos"
VIDEOS_DIR.mkdir(parents=True, exist_ok=True)


def _local_path_from_storage_url(image_url: str) -> Optional[Path]:
    """将 /storage/... 或本机 storage URL 映射到后端文件路径。"""
    path = image_url
    if image_url.startswith(("http://", "https://")):
        parsed = urlparse(image_url)
        if parsed.hostname not in {"localhost", "127.0.0.1", "0.0.0.0"}:
            return None
        path = parsed.path
    if not path.startswith("/storage/"):
        return None
    return BASE_DIR / path.lstrip("/")


def _tokenhub_image(image_url: str) -> dict:
    """将本地图片转为 TokenHub 所需的裸 Base64；公网 URL 直接透传。"""
    local_path = _local_path_from_storage_url(image_url)
    if local_path is None:
        return {"url": image_url}
    if not local_path.exists():
        raise FileNotFoundError(f"本地图片不存在: {local_path}")
    image_bytes = local_path.read_bytes()
    return {"base64": base64.b64encode(image_bytes).decode("ascii")}


def _download_video(video_url: str, prompt: str) -> str:
    """下载临时结果 URL，避免上游视频链接过期。"""
    response = requests.get(video_url, timeout=300, stream=True)
    response.raise_for_status()
    suffix = Path(urlparse(video_url).path).suffix.lower()
    if suffix not in {".mp4", ".mov", ".webm"}:
        suffix = ".mp4"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    prompt_part = "".join(char if char.isalnum() or char in "-_" else "_" for char in prompt[:24]).strip("_")
    filename = f"tokenhub_{timestamp}_{uuid.uuid4().hex[:8]}_{prompt_part}{suffix}" if prompt_part else f"tokenhub_{timestamp}_{uuid.uuid4().hex[:8]}{suffix}"
    target = VIDEOS_DIR / filename
    with target.open("wb") as output:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                output.write(chunk)
    return f"/storage/videos/{filename}"


class GenerateTokenHubHyVideoInput(BaseModel):
    """TokenHub HY-Video-1.5 的输入参数。"""

    prompt: str = Field(description="视频内容描述，最多 200 个 UTF-8 字符。")
    mode: Literal["text", "image"] = Field(default="text", description="text 为文生视频；image 为单图图生视频。")
    image_url: Optional[str] = Field(default=None, description="image 模式必填；支持公网 URL 或本地 /storage/images/... 路径。")
    logo_add: int = Field(default=1, description="是否添加 AI 标识：1 添加（默认），0 关闭（需平台已获授权）。")


@tool("generate_tokenhub_hy_video", args_schema=GenerateTokenHubHyVideoInput)
def generate_tokenhub_hy_video_tool(
    prompt: str,
    mode: Literal["text", "image"] = "text",
    image_url: Optional[str] = None,
    logo_add: int = 1,
) -> str:
    """使用腾讯云 TokenHub 的 HY-Video-1.5 生成视频。

    官方文档：https://cloud.tencent.com/document/product/1823/130081
    价格文档：https://cloud.tencent.com/document/product/1823/130055
    仅支持文生视频和单图图生视频；HY-Video-1.5 当前 TokenHub 接口不支持首尾帧或多参考图。
    价格（2026-08-05）：1.5 积分/次 × 1 元/积分 = 1.50 元/次。官方未承诺固定输出时长，
    因此元/秒应按 1.50 ÷ 实际输出秒数计算；若输出 5 秒约 0.30 元/秒，输出 10 秒约 0.15 元/秒。

    返回下载到本地 /storage/videos/ 的 video_url JSON。
    """
    if not TOKENHUB_API_KEY:
        return json.dumps({"error": "未配置 TOKENHUB_API_KEY（请在 backend/.env 设置，可参考 env.example）"}, ensure_ascii=False)
    if mode == "image" and not image_url:
        return json.dumps({"error": "image 模式必须提供 image_url"}, ensure_ascii=False)

    payload = {"model": TOKENHUB_VIDEO_MODEL, "prompt": prompt, "logo_add": logo_add}
    if image_url:
        payload["image"] = _tokenhub_image(image_url)
    headers = {"Authorization": f"Bearer {TOKENHUB_API_KEY}", "Content-Type": "application/json"}

    try:
        submit = requests.post(f"{TOKENHUB_BASE_URL}/v1/api/video/submit", json=payload, headers=headers, timeout=60)
        if not submit.ok:
            raise RuntimeError(f"提交任务失败: status={submit.status_code}, body={submit.text}")
        task = submit.json()
        task_id = task.get("id")
        if not task_id:
            raise RuntimeError(f"提交响应未包含 id: {json.dumps(task, ensure_ascii=False)}")

        deadline = time.monotonic() + POLL_TIMEOUT_SECONDS
        while time.monotonic() < deadline:
            query = requests.post(
                f"{TOKENHUB_BASE_URL}/v1/api/video/query",
                json={"model": TOKENHUB_VIDEO_MODEL, "id": task_id},
                headers=headers,
                timeout=60,
            )
            if not query.ok:
                raise RuntimeError(f"查询任务失败: status={query.status_code}, body={query.text}")
            result = query.json()
            status = str(result.get("status", "")).lower()
            if status in {"failed", "fail", "error", "cancelled", "canceled"}:
                raise RuntimeError(result.get("error") or result.get("message") or f"任务失败: {json.dumps(result, ensure_ascii=False)}")
            video_url = result.get("data", {}).get("url") if isinstance(result.get("data"), dict) else None
            if status in {"completed", "succeeded", "success", "done"} and video_url:
                local_path = _download_video(video_url, prompt)
                return json.dumps({
                    "video_url": local_path,
                    "original_url": video_url,
                    "local_path": local_path,
                    "task_id": task_id,
                    "provider": "tokenhub",
                    "model": TOKENHUB_VIDEO_MODEL,
                    "mode": mode,
                    "prompt": prompt,
                    "price_yuan_per_request": 1.5,
                    "price_yuan_per_second": "1.50 / 实际输出秒数",
                    "message": "视频已生成并保存到本地",
                }, ensure_ascii=False)
            time.sleep(POLL_INTERVAL_SECONDS)
        raise TimeoutError(f"任务超时: {POLL_TIMEOUT_SECONDS} 秒内未完成（task_id={task_id}）")
    except Exception as error:
        logger.exception("TokenHub HY-Video-1.5 生成失败")
        return json.dumps({"error": f"生成视频时出错: {error}"}, ensure_ascii=False)


def main() -> None:
    """命令行测试入口；使用有效 API Key 会产生 TokenHub 视频生成费用。"""
    parser = argparse.ArgumentParser(description="测试 TokenHub HY-Video-1.5 视频生成 Tool")
    parser.add_argument("--prompt", default="一只小狗在阳光下的草地上奔跑", help="视频提示词")
    parser.add_argument("--mode", choices=("text", "image"), default="text", help="生成模式")
    parser.add_argument("--image-url", help="image 模式使用的公网 URL 或 /storage/images/... 本地路径")
    parser.add_argument("--logo-add", type=int, choices=(0, 1), default=1, help="是否添加 AI 标识：0 否，1 是")
    args = parser.parse_args()

    if args.mode == "image" and not args.image_url:
        parser.error("--mode image 时必须提供 --image-url")

    logging.basicConfig(level=logging.INFO)
    result = generate_tokenhub_hy_video_tool.invoke({
        "prompt": args.prompt,
        "mode": args.mode,
        "image_url": args.image_url,
        "logo_add": args.logo_add,
    })
    print(result)


if __name__ == "__main__":
    main()
