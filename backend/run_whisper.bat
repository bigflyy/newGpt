@echo off
start "" "..\..\whisper.cpp\build\bin\Release\whisper-server.exe" -m "..\..\whisper.cpp\models\ggml-large-v3-q5_0.bin" --vad -vm "..\..\whisper.cpp\models\ggml-silero-v6.2.0.bin" -l ru