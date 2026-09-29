"""Gemini を呼んで判定を得る (issue #46)。API キーは環境変数 GEMINI_API_KEY で渡す。"""

from __future__ import annotations

import time
from typing import Literal

from google import genai
from google.genai import errors, types
from pydantic import BaseModel

from analysis.word_sense.prompt import build_prompt, instructions
from analysis.word_sense.records import CATEGORIES, LlmOutput

RETRIES = 5
BACKOFF_SECONDS = 4.0


class Verdict(BaseModel):
    """構造化出力の形。category は 4 択に限る。"""

    category: Literal["一致", "一部一致", "別の意味で解釈", "無関係"]
    reason: str


class Judge:
    def __init__(self, model_name: str, api_key: str) -> None:
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name
        info = self.client.models.get(model=model_name)
        self.model_label = f"{model_name} ({info.version})" if info.version else model_name
        self.config = types.GenerateContentConfig(
            system_instruction=instructions(),
            response_mime_type="application/json",
            response_schema=Verdict,
            # 関数呼び出しは使わない。既定のままだと呼び出しごとに勧告が出る
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        )

    def judge(self, title: str, description: str, document_text: str) -> LlmOutput:
        """一時的な失敗 (レート制限、サーバ側のエラー) は間を空けて試し直す。"""
        prompt = build_prompt(title, description, document_text)
        for attempt in range(RETRIES):
            try:
                response = self.client.models.generate_content(
                    model=self.model_name, contents=prompt, config=self.config
                )
            except errors.APIError as error:
                if attempt == RETRIES - 1 or error.code not in (429, 500, 503):
                    raise
                time.sleep(BACKOFF_SECONDS * (2**attempt))
                continue
            return self._parse(response)
        raise RuntimeError("到達しない")

    def _parse(self, response: types.GenerateContentResponse) -> LlmOutput:
        raw = response.text or ""
        parsed = response.parsed
        if not isinstance(parsed, Verdict):
            raise ValueError(f"応答を判定として読めませんでした。応答: {raw[:200]}")
        if parsed.category not in CATEGORIES:
            raise ValueError(f"想定外の分類です: {parsed.category}")
        reason = parsed.reason if parsed.category == "別の意味で解釈" else ""
        return LlmOutput(model=self.model_label, category=parsed.category, reason=reason, raw=raw)
