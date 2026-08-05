"""fal.ai Kling O3 Standard 视频生成 Tool。"""
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

KLING_ENDPOINTS = {
    "text": "fal-ai/kling-video/o3/standard/text-to-video",
    "image": "fal-ai/kling-video/o3/standard/image-to-video",
    "reference": "fal-ai/kling-video/o3/standard/reference-to-video",
}


class GenerateFalKlingO3StandardVideoInput(BaseModel):
    """Kling O3 Standard 的输入参数。"""

    prompt: str = Field(description="视频内容描述；reference 模式可用 @Image1、@Image2 引用参考图。")
    mode: Literal["text", "image", "reference"] = Field(default="text", description="文生、单图/首尾帧图生或参考图生。")
    image_urls: list[str] = Field(default_factory=list, description="image 模式恰好 1 张首帧；reference 模式 1–4 张参考图。")
    end_image_url: Optional[str] = Field(default=None, description="image/reference 模式可选尾帧。")
    duration: Literal["3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15"] = Field(default="5", description="视频时长（秒）。")
    aspect_ratio: Literal["16:9", "9:16", "1:1"] = Field(default="16:9", description="输出宽高比。")
    generate_audio: bool = Field(default=False, description="是否生成原生音频。")


@tool("generate_fal_kling_o3_standard_video", args_schema=GenerateFalKlingO3StandardVideoInput)
def generate_fal_kling_o3_standard_video_tool(
    prompt: str,
    mode: Literal["text", "image", "reference"] = "text",
    image_urls: Optional[list[str]] = None,
    end_image_url: Optional[str] = None,
    duration: Literal["3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15"] = "5",
    aspect_ratio: Literal["16:9", "9:16", "1:1"] = "16:9",
    generate_audio: bool = False,
) -> str:
    """通过 fal.ai 调用 Kling O3 Standard 生成视频。

    官方文档：
    - 文生：https://fal.ai/docs/model-api-reference/video-generation-api/kling-video-o3-standard-text-to-video
    - 图生/首尾帧：https://fal.ai/models/fal-ai/kling-video/o3/standard/image-to-video/api
    - 参考图生：https://fal.ai/docs/model-api-reference/video-generation-api/kling-video-o3-standard-reference-to-video
    - fal Python SDK：https://fal.ai/docs/api-reference/client-libraries/python/fal_client
    支持文生、单图/首尾帧图生、1–4 张参考图生；视频时长为 3–15 秒。
    价格（2026-08-05，fal 官方美元报价）：无音频 $0.084/秒，有音频 $0.112/秒。
    按 1 USD ≈ 6.77 元估算：无音频 ≈ 0.57 元/秒，有音频 ≈ 0.76 元/秒；
    汇率会波动，实际换算值由 FAL_USD_CNY_RATE 配置决定，最终以 fal 账单为准。

    返回下载到本地 /storage/videos/ 的 video_url JSON。
    """
    image_urls = image_urls or []
    if mode == "text" and (image_urls or end_image_url):
        return json.dumps({"error": "text 模式不接受 image_urls 或 end_image_url"}, ensure_ascii=False)
    if mode == "image" and len(image_urls) != 1:
        return json.dumps({"error": "image 模式需要恰好 1 张首帧图片"}, ensure_ascii=False)
    if mode == "reference" and not 1 <= len(image_urls) <= 4:
        return json.dumps({"error": "reference 模式需要 1 至 4 张参考图"}, ensure_ascii=False)

    try:
        client = create_fal_client()
        payload = {"prompt": prompt, "duration": duration, "aspect_ratio": aspect_ratio, "generate_audio": generate_audio}
        if mode == "image":
            payload["image_url"] = resolve_input_url(image_urls[0], client)
        elif mode == "reference":
            payload["image_urls"] = [resolve_input_url(image_url, client) for image_url in image_urls]
        if end_image_url:
            payload["end_image_url"] = resolve_input_url(end_image_url, client)
        result = generate_fal_video(KLING_ENDPOINTS[mode], payload, prompt, "klingo3")
        price_usd = 0.112 if generate_audio else 0.084
        return json.dumps({
            **result,
            "provider": "fal.ai", "model": "Kling O3 Standard", "mode": mode,
            "endpoint": KLING_ENDPOINTS[mode], "prompt": prompt,
            "price_usd_per_second": price_usd, "price_yuan_per_second_estimate": round(price_usd * FAL_USD_CNY_RATE, 4),
            "usd_cny_rate_used": FAL_USD_CNY_RATE, "message": "视频已生成并保存到本地",
        }, ensure_ascii=False)
    except Exception as error:
        logging.exception("fal Kling O3 Standard 生成失败")
        return json.dumps({"error": f"生成视频时出错: {error}"}, ensure_ascii=False)


def main() -> None:
    """命令行测试入口；使用有效 FAL_KEY 会产生 fal 视频生成费用。"""
    parser = argparse.ArgumentParser(description="测试 fal Kling O3 Standard 视频 Tool")
    parser.add_argument("--prompt", default="一只小狗在阳光下的草地上奔跑", help="视频提示词")
    parser.add_argument("--mode", choices=("text", "image", "reference"), default="text")
    parser.add_argument("--image-url", dest="image_urls", action="append", help="首帧或参考图；可重复传入 reference 图片")
    parser.add_argument("--end-image-url", help="image/reference 模式可选尾帧")
    parser.add_argument("--duration", choices=tuple(str(value) for value in range(3, 16)), default="5")
    parser.add_argument("--aspect-ratio", choices=("16:9", "9:16", "1:1"), default="16:9")
    parser.add_argument("--generate-audio", action="store_true")
    args = parser.parse_args()
    image_urls = args.image_urls or []
    if args.mode == "image" and len(image_urls) != 1:
        parser.error("--mode image 时必须恰好提供一个 --image-url")
    if args.mode == "reference" and not 1 <= len(image_urls) <= 4:
        parser.error("--mode reference 时必须提供 1 至 4 个 --image-url")
    if args.mode == "text" and (image_urls or args.end_image_url):
        parser.error("--mode text 不接受图片参数")
    logging.basicConfig(level=logging.INFO)
    print(generate_fal_kling_o3_standard_video_tool.invoke({
        "prompt": args.prompt, "mode": args.mode, "image_urls": image_urls, "end_image_url": args.end_image_url,
        "duration": args.duration, "aspect_ratio": args.aspect_ratio, "generate_audio": args.generate_audio,
    }))


if __name__ == "__main__":
    main()
