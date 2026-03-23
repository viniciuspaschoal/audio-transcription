🎙️ Audio Meeting Transcriber (Whisper GUI + CLI)

A Python application for automatic transcription of in-person meeting audio, powered by Whisper (faster-whisper).

Designed to be easy for daily use (via GUI) and powerful for automation (via CLI).

✨ Features
🖥️ User-friendly graphical interface (Tkinter)
💻 CLI mode for automation and batch processing
📂 Multi-file import support
📜 Real-time logging (console-style)
🧠 High-accuracy transcription with Whisper
⚡ CPU and GPU (CUDA) support
🔇 Silence filtering (VAD – Voice Activity Detection)
📁 Multiple output formats:
Clean .txt
.txt with timestamps
Structured .json
🧪 Built-in unit tests
🎯 Use Case

Perfect for:

🏢 In-person meetings
📝 Automatic meeting notes
📊 Technical discussion records
🤖 Foundation for internal automation systems
🧱 Tech Stack
Python 3.10+
faster-whisper
Tkinter (native GUI)
ffmpeg (audio processing)
📦 Setup
Clone the repository
git clone https://github.com/your-username/your-repo.git
cd your-repo
Install dependencies
pip install faster-whisper
Install FFmpeg

Required for .mp3, .m4a, .ogg, etc.

Windows: https://ffmpeg.org/download.html
Linux:
sudo apt install ffmpeg
▶️ Usage
🖥️ GUI (Recommended)
python app_transcricao.py

or

python app_transcricao.py --gui
GUI Features:
File selection
Model selection
GPU configuration
Live logs during transcription
Output directory selection
💻 CLI
Specific files
python app_transcricao.py --cli -i audio1.wav audio2.mp3 -o output/
Using glob pattern
python app_transcricao.py --cli -g "audios/*.mp3"
With GPU
python app_transcricao.py --cli -i audio.wav --cuda --compute float16
⚙️ CLI Options
Argument	Description
-i	Input files
-g	Glob pattern
-o	Output directory
-m	Model (tiny → large-v3)
--cuda	Enable GPU
--compute	Precision type
--no-vad	Disable VAD
--vad-ms	Minimum silence duration
📁 Output

For each processed file:

audio_transcription.txt
audio_with_timestamps.txt
audio_transcription.json
JSON Example:
[
  {
    "inicio": 0.0,
    "fim": 2.5,
    "texto": "hello world"
  }
]
🧪 Tests

Run tests without loading the real model:

python app_transcricao.py --test
⚡ Performance Tips
tiny/base → faster (lower accuracy)
small/medium → best balance
large-v3 → highest accuracy (heavier)
int8 → lighter
float16 → better performance on GPU
🛣️ Roadmap
 Web interface (FastAPI / Flask)
 Speaker diarization (who spoke)
 Automatic summarization (AI)
 Dashboard with history
 Database integration
⚠️ Notes
GPU significantly improves performance
FFmpeg is required for compressed formats
Long audio files may consume high RAM
