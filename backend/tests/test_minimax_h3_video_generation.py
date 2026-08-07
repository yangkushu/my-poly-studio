import json
import sys
from unittest.mock import Mock, patch

sys.path.insert(0, "backend")

from app.tools import minimax_h3_video_generation as h3


def _response(payload):
    response = Mock()
    response.ok = True
    response.json.return_value = payload
    response.text = json.dumps(payload)
    return response


def test_text_generation_submits_official_v2_payload_and_downloads_result():
    post = Mock(return_value=_response({"task_id": "task-123"}))
    get = Mock(return_value=_response({
        "task": {
            "id": "task-123", "status": "succeeded", "content": {"url": "https://cdn.example/video.mp4"},
            "resolution": "2K", "duration": 5, "ratio": "16:9", "usage": {"output_seconds": 5},
        }
    }))
    with patch.object(h3, "MINIMAX_API_KEY", "test-key"), \
         patch.object(h3.requests, "post", post), \
         patch.object(h3.requests, "get", get), \
         patch.object(h3, "_download_video", return_value="/storage/videos/minimax_h3.mp4"):
        result = json.loads(h3.generate_minimax_h3_video_tool.invoke({
            "prompt": "a dog runs across a sunny meadow", "duration": 5, "resolution": "2K", "aspect_ratio": "16:9",
        }))

    assert result["video_url"] == "/storage/videos/minimax_h3.mp4"
    assert result["task_id"] == "task-123"
    assert post.call_args.args[0] == "https://api.minimaxi.com/v2/video_generation"
    assert post.call_args.kwargs["headers"]["Authorization"] == "Bearer test-key"
    assert post.call_args.kwargs["json"] == {
        "model": "MiniMax-H3",
        "content": [{"type": "text", "text": "a dog runs across a sunny meadow"}],
        "resolution": "2K", "duration": 5, "ratio": "16:9", "aigc_watermark": False,
    }
    assert get.call_args.args[0] == "https://api.minimaxi.com/v2/query/video_generation/task-123"


def test_image_mode_sets_first_and_last_frame_and_forces_adaptive_ratio():
    post = Mock(return_value=_response({"task_id": "task-456"}))
    get = Mock(return_value=_response({"task": {
        "status": "succeeded", "content": {"url": "https://cdn.example/video.mp4"},
    }}))
    with patch.object(h3, "MINIMAX_API_KEY", "test-key"), \
         patch.object(h3.requests, "post", post), \
         patch.object(h3.requests, "get", get), \
         patch.object(h3, "_download_video", return_value="/storage/videos/minimax_h3.mp4"):
        result = json.loads(h3.generate_minimax_h3_video_tool.invoke({
            "prompt": "turn day into night", "mode": "image", "start_image_url": "https://example.com/first.png",
            "end_image_url": "https://example.com/last.png", "aspect_ratio": "16:9",
        }))

    payload = post.call_args.kwargs["json"]
    assert payload["ratio"] == "adaptive"
    assert payload["content"][1] == {
        "type": "image_url", "image_url": {"url": "https://example.com/first.png"}, "role": "first_frame",
    }
    assert payload["content"][2]["role"] == "last_frame"
    assert result["ratio"] == "adaptive"


def test_reference_mode_rejects_frame_images_before_submitting():
    with patch.object(h3, "MINIMAX_API_KEY", "test-key"), patch.object(h3.requests, "post") as post:
        result = json.loads(h3.generate_minimax_h3_video_tool.invoke({
            "prompt": "make a film", "mode": "reference", "start_image_url": "https://example.com/frame.png",
            "reference_audio_urls": ["https://example.com/music.mp3"],
        }))

    assert result == {"error": "reference 模式不能与首帧或尾帧图片混用"}
    post.assert_not_called()


if __name__ == "__main__":
    test_text_generation_submits_official_v2_payload_and_downloads_result()
    test_image_mode_sets_first_and_last_frame_and_forces_adaptive_ratio()
    test_reference_mode_rejects_frame_images_before_submitting()
