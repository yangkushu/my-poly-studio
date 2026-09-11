import importlib
import json
import unittest
from pathlib import Path
from unittest.mock import Mock, patch


MODULE_NAME = "app.tools.doubao_omni_understanding"


def _module():
    return importlib.import_module(MODULE_NAME)


class DoubaoOmniUnderstandingTests(unittest.TestCase):
    def test_builds_remote_audio_content(self):
        module = _module()
        content, file_id = module._build_media_content("https://cdn.example.com/sample.wav", "audio")
        self.assertEqual(content["type"], "input_audio")
        self.assertEqual(content["audio_url"], "https://cdn.example.com/sample.wav")
        self.assertIsNone(file_id)

    def test_builds_remote_image_content(self):
        module = _module()
        content, file_id = module._build_media_content("https://cdn.example.com/sample.png", "image")
        self.assertEqual(content["type"], "input_image")
        self.assertEqual(content["image_url"], "https://cdn.example.com/sample.png")
        self.assertIsNone(file_id)

    def test_uploads_local_storage_audio_and_uses_file_id(self):
        module = _module()
        audio_file = module.BASE_DIR / "storage" / "audios" / "sample.wav"
        audio_file.parent.mkdir(parents=True, exist_ok=True)
        upload_response = Mock()
        upload_response.json.return_value = {"id": "file-audio-123"}
        try:
            audio_file.write_bytes(b"fake wav")
            with patch.object(module, "VOLCANO_API_KEY", "test-key"), patch.object(
                module.requests, "post", return_value=upload_response
            ) as post:
                content, file_id = module._build_media_content("/storage/audios/sample.wav", "audio")
        finally:
            audio_file.unlink(missing_ok=True)

        self.assertEqual(content, {"type": "input_audio", "file_id": "file-audio-123"})
        self.assertEqual(file_id, "file-audio-123")
        self.assertEqual(post.call_args.args[0], "https://ark.cn-beijing.volces.com/api/v3/files")
        self.assertEqual(post.call_args.kwargs["data"], {"purpose": "user_data"})
        self.assertEqual(post.call_args.kwargs["headers"]["Authorization"], "Bearer test-key")

    def test_calls_responses_api_and_extracts_output_text(self):
        module = _module()
        response = Mock()
        response.json.return_value = {
            "output": [
                {
                    "type": "message",
                    "content": [{"type": "output_text", "text": "识别到音乐。"}],
                }
            ]
        }
        with patch.object(module.requests, "post", return_value=response) as post:
            text = module._call_doubao([{"role": "user", "content": []}])

        self.assertEqual(text, "识别到音乐。")
        self.assertEqual(post.call_args.args[0], "https://ark.cn-beijing.volces.com/api/v3/responses")
        self.assertEqual(post.call_args.kwargs["json"]["model"], module.DOUBAO_OMNI_MODEL)
        self.assertFalse(post.call_args.kwargs["json"]["store"])

    def test_tool_returns_existing_json_shape(self):
        module = _module()
        with patch.object(module, "VOLCANO_API_KEY", "test-key"), patch.object(
            module,
            "_call_doubao",
            return_value="检测到舒缓音乐和一位说话者。",
        ):
            result = module.doubao_omni_understand_tool.invoke(
                {
                    "media_path": "https://cdn.example.com/sample.mp3",
                    "question": "这段音频有什么？",
                }
            )

        payload = json.loads(result)
        self.assertEqual(payload["text_response"], "检测到舒缓音乐和一位说话者。")
        self.assertEqual(payload["media_type"], "audio")
        self.assertEqual(payload["provider"], "volcano-doubao")

    def test_tool_cleans_up_uploaded_local_media(self):
        module = _module()
        audio_file = module.BASE_DIR / "storage" / "audios" / "cleanup-sample.wav"
        audio_file.parent.mkdir(parents=True, exist_ok=True)
        try:
            audio_file.write_bytes(b"fake wav")
            with patch.object(module, "VOLCANO_API_KEY", "test-key"), patch.object(
                module, "_upload_file_to_ark", return_value="file-cleanup-123"
            ), patch.object(
                module, "_call_doubao", return_value="检测到一段语音。"
            ), patch.object(module, "_delete_ark_file") as delete:
                result = module.doubao_omni_understand_tool.invoke(
                    {
                        "media_path": "/storage/audios/cleanup-sample.wav",
                        "question": "分析音频",
                        "media_type": "audio",
                    }
                )
        finally:
            audio_file.unlink(missing_ok=True)

        self.assertEqual(json.loads(result)["text_response"], "检测到一段语音。")
        delete.assert_called_once_with("file-cleanup-123")
