# N3 API 測試語料

測試語料由 `generate_test_corpus.py` 即時產生，不保存訪談音訊或逐字稿。
涵蓋 WAV／FLAC／MP3、mono／stereo、8／16／44.1／48 kHz、1／3／6／12 秒，
以及 invalid／empty cases；`manifest.json` 會記錄 SHA-256 與預期狀態。

    Crab/.venv/bin/python -m Crab.api.generate_test_corpus \
      --output-dir /tmp/n3-api-corpus --require-compressed

產物刻意放在 `/tmp`，避免把二進位測試音訊加入 Git。
