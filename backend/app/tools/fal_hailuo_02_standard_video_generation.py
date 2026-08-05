"""fal.ai MiniMax Hailuo 02 Standard 视频生成 Tool。"""
import argparse
import json
import logging
from typing import Literal, Optional

from langchain_core.tools import tool
from pydantic import BaseModel, Field

try:
    from app.tools.fal_video_generation import FAL_USD_CNY_RATE, create_fal_client, generate_fal_video, resolve_input_url
except ModuleNotFoundError:  # 支持 cd backend 后直接执行 python app/tools/<tool>.py
    from fal_video_generation import FAL_USD_CNY_RATE, create_fal_client, generate_fal_video, resolve_input_url

HAILUO_ENDPOINTS = {
    "text": "fal-ai/minimax/hailuo-02/standard/text-to-video",
    "image": "fal-ai/minimax/hailuo-02/standard/image-to-video",
}


class GenerateFalHailuo02StandardVideoInput(BaseModel):
    """Hailuo 02 Standard 的输入参数。"""

    prompt: str = Field(description="视频内容描述。")
    mode: Literal["text", "image"] = Field(default="text", description="text 为文生视频；image 为单图/首尾帧图生视频。")
    image_url: Optional[str] = Field(default=None, description="image 模式必填；支持公网 URL、data URI 或本地 /storage/... 路径。")
    end_image_url: Optional[str] = Field(default=None, description="image 模式可选尾帧；支持公网 URL、data URI 或本地 /storage/... 路径。")
    duration: Literal["6", "10"] = Field(default="6", description="视频时长（秒）。")
    resolution: Literal["512P", "768P"] = Field(default="768P", description="输出分辨率。")
    prompt_optimizer: bool = Field(default=True, description="是否启用模型提示词优化。")


@tool("generate_fal_hailuo_02_standard_video", args_schema=GenerateFalHailuo02StandardVideoInput)
def generate_fal_hailuo_02_standard_video_tool(
    prompt: str,
    mode: Literal["text", "image"] = "text",
    image_url: Optional[str] = None,
    end_image_url: Optional[str] = None,
    duration: Literal["6", "10"] = "6",
    resolution: Literal["512P", "768P"] = "768P",
    prompt_optimizer: bool = True,
) -> str:
    """通过 fal.ai 调用 MiniMax Hailuo 02 Standard（768P/512P）生成视频。

    官方文档：
    - 文生：https://fal.ai/models/fal-ai/minimax/hailuo-02/standard/text-to-video/api
    - 图生/首尾帧：https://fal.ai/models/fal-ai/minimax/hailuo-02/standard/image-to-video/api
    - fal Python SDK：https://fal.ai/docs/api-reference/client-libraries/python/fal_client
    支持文生，或单图图生（可选尾帧）；时长为 6 或 10 秒，分辨率为 512P 或 768P。
    价格（2026-08-05，fal 官方美元报价）：512P 为 $0.017/秒，768P 为 $0.045/秒。
    按 1 USD ≈ 6.77 元估算：512P ≈ 0.12 元/秒，768P ≈ 0.30 元/秒；
    汇率会波动，实际换算值由 FAL_USD_CNY_RATE 配置决定，最终以 fal 账单为准。

    返回下载到本地 /storage/videos/ 的 video_url JSON。
    """
    if mode == "image" and not image_url:
        return json.dumps({"error": "image 模式必须提供 image_url"}, ensure_ascii=False)
    if mode == "text" and (image_url or end_image_url):
        return json.dumps({"error": "text 模式不接受 image_url 或 end_image_url"}, ensure_ascii=False)

    try:
        client = create_fal_client()
        payload = {"prompt": prompt, "duration": duration, "resolution": resolution, "prompt_optimizer": prompt_optimizer}
        if mode == "image":
            payload["image_url"] = resolve_input_url(image_url, client)
            if end_image_url:
                payload["end_image_url"] = resolve_input_url(end_image_url, client)
        result = generate_fal_video(HAILUO_ENDPOINTS[mode], payload, prompt, "hailuo02")
        price_usd = 0.017 if resolution == "512P" else 0.045
        return json.dumps({
            **result,
            "provider": "fal.ai", "model": "MiniMax Hailuo 02 Standard", "mode": mode,
            "endpoint": HAILUO_ENDPOINTS[mode], "prompt": prompt, "resolution": resolution,
            "price_usd_per_second": price_usd, "price_yuan_per_second_estimate": round(price_usd * FAL_USD_CNY_RATE, 4),
            "usd_cny_rate_used": FAL_USD_CNY_RATE, "message": "视频已生成并保存到本地",
        }, ensure_ascii=False)
    except Exception as error:
        logging.exception("fal Hailuo 02 Standard 生成失败")
        return json.dumps({"error": f"生成视频时出错: {error}"}, ensure_ascii=False)


def main() -> None:
    """命令行测试入口；使用有效 FAL_KEY 会产生 fal 视频生成费用。"""
    parser = argparse.ArgumentParser(description="测试 fal MiniMax Hailuo 02 Standard 视频 Tool")
    parser.add_argument("--prompt", default="一只小狗在阳光下的草地上奔跑", help="视频提示词")
    parser.add_argument("--mode", choices=("text", "image"), default="text")
    parser.add_argument("--image-url", help="image 模式的首帧")
    parser.add_argument("--end-image-url", help="image 模式可选尾帧")
    parser.add_argument("--duration", choices=("6", "10"), default="6")
    parser.add_argument("--resolution", choices=("512P", "768P"), default="768P")
    parser.add_argument("--no-prompt-optimizer", action="store_true")
    args = parser.parse_args()
    if args.mode == "image" and not args.image_url:
        parser.error("--mode image 时必须提供 --image-url")
    logging.basicConfig(level=logging.INFO)
    print(generate_fal_hailuo_02_standard_video_tool.invoke({
        "prompt": args.prompt, "mode": args.mode, "image_url": args.image_url, "end_image_url": args.end_image_url,
        "duration": args.duration, "resolution": args.resolution, "prompt_optimizer": not args.no_prompt_optimizer,
    }))


if __name__ == "__main__":
    main()
