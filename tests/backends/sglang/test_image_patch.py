# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[3]
_SGLANG_PATCH = _REPO_ROOT / "docker" / "patch" / "latest" / "sglang.patch"


def test_runai_processor_resolves_model_name_uri():
    patch = _SGLANG_PATCH.read_text()
    processor_diff = patch.split(
        "diff --git a/python/sglang/srt/utils/hf_transformers/processor.py ",
        maxsplit=1,
    )[1]
    processor_diff = processor_diff.split("\ndiff --git ", maxsplit=1)[0]
    normalized_diff = processor_diff.replace("\n \n", "\n\n")

    expected_hunk = """@@ -153,6 +153,8 @@ def get_processor(

     revision = kwargs.pop("revision", tokenizer_revision)
     tokenizer_name = resolve_runai_obj_uri(tokenizer_name)
+    if model_name is not None:
+        model_name = resolve_runai_obj_uri(model_name)

     if is_mistral_model(tokenizer_name):
         config = load_mistral_config("""
    assert expected_hunk in normalized_diff
