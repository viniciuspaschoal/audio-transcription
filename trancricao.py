# App de Transcrição – GUI (Tkinter) + CLI (fallback)
# -----------------------------------------------------------------
# Atende ao pedido de:
# 1) **Botão para importar arquivos** (File Dialog)
# 2) **Caixa estilo console** rolando ao vivo enquanto transcreve
# 3) Mantém modo **CLI** como alternativa (sem Streamlit)
# 4) Inclui **testes** (`--test`) sem tocar no modelo real
#
# Execução GUI (recomendada para IDE):
#   python app_transcricao.py --gui
# (Se omitir flags, a GUI é o padrão quando possível.)
#
# Execução CLI:
#   python app_transcricao.py --cli -i audio1.wav audio2.m4a -o saidas/ -m small --cuda --compute float16
#   python app_transcricao.py --cli -g "entrada/*.mp3"
#
# Testes:
#   python app_transcricao.py --test
#
# Dependências:
#   pip install faster-whisper
#   (ffmpeg no PATH para mp3/m4a/ogg)

from __future__ import annotations
import argparse
import glob
import json
import os
import sys
import threading
import time
from dataclasses import dataclass
from typing import Iterable, List, Tuple

try:
    from faster_whisper import WhisperModel
except Exception as e:
    print("\n❌ Erro ao importar faster-whisper.\n   Instale:  pip install faster-whisper\n   Detalhe:", e, file=sys.stderr)
    sys.exit(2)

# ========================= Núcleo reutilizável =========================

@dataclass
class Trecho:
    inicio: float
    fim: float
    texto: str


def carregar_modelo(nome_modelo: str, usar_gpu: bool, compute_type: str) -> WhisperModel:
    device = "cuda" if usar_gpu else "cpu"
    return WhisperModel(nome_modelo, device=device, compute_type=compute_type)


def transcrever_arquivo(model: WhisperModel, path: str, vad: bool, vad_ms: int,
                        on_segment=None) -> Tuple[str, str, List[dict]]:
    """
    Transcreve um arquivo e retorna (texto_limpo, texto_com_tempo, linhas_json).
    Se `on_segment` for fornecido, é chamado a cada segmento encontrado (para UI vivo).
    """
    segments, info = model.transcribe(
        path,
        vad_filter=vad,
        vad_parameters=dict(min_silence_duration_ms=vad_ms),
    )

    texto_limpo_parts: List[str] = []
    texto_com_tempo_lines: List[str] = []
    linhas_json: List[dict] = []

    for seg in segments:
        txt = (seg.text or "").strip()
        if on_segment:
            try:
                on_segment(seg.start, seg.end, txt)
            except Exception:
                pass
        if txt:
            texto_limpo_parts.append(txt)
            texto_com_tempo_lines.append(f"[{seg.start:.2f}s - {seg.end:.2f}s] {txt}")
            linhas_json.append({
                "inicio": float(seg.start),
                "fim": float(seg.end),
                "texto": txt,
            })

    texto_limpo = " ".join(texto_limpo_parts).strip()
    texto_com_tempo = "\n".join(texto_com_tempo_lines)
    return texto_limpo, texto_com_tempo, linhas_json


def salvar_saidas(base_out: str, texto_limpo: str, texto_tempo: str, linhas_json: List[dict],
                  salvar_txt_limpo: bool = True, salvar_txt_tempo: bool = True, salvar_json: bool = True) -> None:
    os.makedirs(os.path.dirname(base_out) or ".", exist_ok=True)
    if salvar_txt_limpo:
        with open(base_out + "_transcricao.txt", "w", encoding="utf-8") as f:
            f.write(texto_limpo)
    if salvar_txt_tempo:
        with open(base_out + "_com_timestamps.txt", "w", encoding="utf-8") as f:
            f.write(texto_tempo)
    if salvar_json:
        with open(base_out + "_transcricao.json", "w", encoding="utf-8") as f:
            json.dump(linhas_json, f, ensure_ascii=False, indent=2)


# ========================= GUI (Tkinter) =========================

def run_gui():
    # Importa tkinter apenas no modo GUI
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox
    from tkinter.scrolledtext import ScrolledText

    class App(tk.Tk):
        def __init__(self):
            super().__init__()
            self.title("Transcrição de Áudio – Whisper (GUI)")
            self.geometry("900x600")

            # Estado
            self.selected_files: List[str] = []
            self.model_name = tk.StringVar(value="small")
            self.use_cuda = tk.BooleanVar(value=False)
            self.compute_type = tk.StringVar(value="int8")
            self.vad_enabled = tk.BooleanVar(value=True)
            self.vad_ms = tk.IntVar(value=200)
            self.output_dir = tk.StringVar(value=os.getcwd())

            self._build_widgets()
            self.model: WhisperModel | None = None
            self.worker: threading.Thread | None = None

        # ---------- UI ----------
        def _build_widgets(self):
            top = ttk.Frame(self)
            top.pack(fill=tk.X, padx=10, pady=8)

            # Seleção de arquivos
            ttk.Button(top, text="📂 Selecionar arquivos...", command=self.on_pick_files).pack(side=tk.LEFT)
            ttk.Label(top, textvariable=tk.StringVar(value="  (suporta wav/mp3/m4a/flac/ogg)"), foreground="#666").pack(side=tk.LEFT)

            # Lista de selecionados
            mid = ttk.Frame(self)
            mid.pack(fill=tk.X, padx=10, pady=4)
            ttk.Label(mid, text="Arquivos selecionados:").pack(anchor=tk.W)
            self.listbox = tk.Listbox(mid, height=5)
            self.listbox.pack(fill=tk.X)

            # Opções
            opt = ttk.LabelFrame(self, text="Opções")
            opt.pack(fill=tk.X, padx=10, pady=8)

            row1 = ttk.Frame(opt); row1.pack(fill=tk.X, pady=4)
            ttk.Label(row1, text="Modelo:").pack(side=tk.LEFT)
            ttk.Combobox(row1, textvariable=self.model_name, values=["tiny","base","small","medium","large-v3"], width=10, state="readonly").pack(side=tk.LEFT, padx=6)
            ttk.Checkbutton(row1, text="Usar GPU (CUDA)", variable=self.use_cuda).pack(side=tk.LEFT, padx=12)
            ttk.Label(row1, text="compute_type:").pack(side=tk.LEFT, padx=12)
            ttk.Combobox(row1, textvariable=self.compute_type, values=["int8","int8_float16","float16","float32"], width=12, state="readonly").pack(side=tk.LEFT)

            row2 = ttk.Frame(opt); row2.pack(fill=tk.X, pady=4)
            ttk.Checkbutton(row2, text="Filtrar silêncio (VAD)", variable=self.vad_enabled).pack(side=tk.LEFT)
            ttk.Label(row2, text="Silêncio mínimo (ms):").pack(side=tk.LEFT, padx=12)
            ttk.Spinbox(row2, from_=100, to=2000, textvariable=self.vad_ms, increment=50, width=8).pack(side=tk.LEFT)
            ttk.Label(row2, text="Saída:").pack(side=tk.LEFT, padx=12)
            out_entry = ttk.Entry(row2, textvariable=self.output_dir, width=40)
            out_entry.pack(side=tk.LEFT)
            ttk.Button(row2, text="📁", width=3, command=self.on_pick_outdir).pack(side=tk.LEFT, padx=4)

            # Ações
            act = ttk.Frame(self)
            act.pack(fill=tk.X, padx=10, pady=8)
            self.btn_start = ttk.Button(act, text="▶️ Transcrever", command=self.on_start)
            self.btn_start.pack(side=tk.LEFT)
            self.prog = ttk.Progressbar(act, mode="indeterminate")
            self.prog.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=10)

            # Console
            con = ttk.LabelFrame(self, text="Console")
            con.pack(fill=tk.BOTH, expand=True, padx=10, pady=8)
            self.console = ScrolledText(con, height=18)
            self.console.pack(fill=tk.BOTH, expand=True)

        # ---------- Handlers ----------
        def on_pick_files(self):
            paths = filedialog.askopenfilenames(title="Selecione arquivos de áudio",
                                                filetypes=[("Áudio", "*.wav *.mp3 *.m4a *.flac *.ogg"), ("Todos", "*.*")])
            if not paths:
                return
            self.selected_files = list(paths)
            self.listbox.delete(0, tk.END)
            for p in self.selected_files:
                self.listbox.insert(tk.END, p)
            self.log(f"Selecionados {len(self.selected_files)} arquivo(s).")

        def on_pick_outdir(self):
            d = filedialog.askdirectory(title="Escolha o diretório de saída", initialdir=self.output_dir.get())
            if d:
                self.output_dir.set(d)

        def on_start(self):
            if not self.selected_files:
                messagebox.showinfo("Info", "Selecione ao menos um arquivo.")
                return
            if self.worker and self.worker.is_alive():
                messagebox.showinfo("Em andamento", "Já existe uma transcrição em andamento.")
                return
            self.prog.start(10)
            self.btn_start.configure(state=tk.DISABLED)
            self.worker = threading.Thread(target=self._worker_run, daemon=True)
            self.worker.start()

        # ---------- Lógica de transcrição com logs ao vivo ----------
        def _worker_run(self):
            t0 = time.time()
            # Carrega modelo (uma vez)
            try:
                self.log(f"Carregando modelo {self.model_name.get()} (cuda={self.use_cuda.get()}, compute={self.compute_type.get()})...")
                self.model = carregar_modelo(self.model_name.get(), self.use_cuda.get(), self.compute_type.get())
            except Exception as e:
                self.log(f"Erro ao carregar modelo: {e}")
                self.finish()
                return

            total_ok = 0
            for path in self.selected_files:
                base_name = os.path.splitext(os.path.basename(path))[0]
                base_out = os.path.join(self.output_dir.get(), base_name)
                self.log(f"\n🔊 Transcrevendo: {path}")
                t1 = time.time()

                def on_seg(start, end, txt):
                    self.log(f"[{start:.2f}s - {end:.2f}s] {txt}")

                try:
                    texto_limpo, texto_tempo, linhas_json = transcrever_arquivo(
                        self.model, path, self.vad_enabled.get(), int(self.vad_ms.get()), on_segment=on_seg
                    )
                except Exception as e:
                    self.log(f"❌ Falha na transcrição de '{path}': {e}")
                    continue

                try:
                    salvar_saidas(base_out, texto_limpo, texto_tempo, linhas_json)
                except Exception as e:
                    self.log(f"❌ Falha ao salvar saídas de '{path}': {e}")
                    continue

                dt = time.time() - t1
                self.log(f"✅ Concluído em {dt:.2f}s → {base_out}_transcricao.txt / _com_timestamps.txt / _transcricao.json")
                total_ok += 1

            self.log(f"\nResumo: {total_ok}/{len(self.selected_files)} arquivo(s) processados.")
            self.finish()

        def log(self, msg: str):
            def _append():
                self.console.insert("end", msg + "\n")
                self.console.see("end")
            self.console.after(0, _append)

        def finish(self):
            def _done():
                self.prog.stop()
                self.btn_start.configure(state=tk.NORMAL)
            self.after(0, _done)

    # Inicia app
    App().mainloop()


# ========================= CLI =========================

def parse_args(argv: List[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Transcrição de áudio – GUI (Tkinter) + CLI (sem Streamlit)")
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--gui", action="store_true", help="Inicia a interface gráfica (padrão)")
    mode.add_argument("--cli", action="store_true", help="Executa em modo linha de comando")

    src = p.add_mutually_exclusive_group()
    src.add_argument("-i", "--inputs", nargs="*", help="Arquivos de áudio (CLI)")
    src.add_argument("-g", "--glob", help="Padrão glob (CLI), ex.: 'pasta/*.mp3'")

    p.add_argument("-o", "--outdir", default=".", help="Diretório de saída (CLI) (default: .)")
    p.add_argument("-m", "--model", default="small", choices=["tiny","base","small","medium","large-v3"], help="Modelo (CLI)")
    p.add_argument("--cuda", action="store_true", help="Usar GPU (CLI)")
    p.add_argument("--compute", default="int8", choices=["int8","int8_float16","float16","float32"], help="compute_type (CLI)")
    p.add_argument("--no-vad", dest="vad", action="store_false", help="Desativar VAD (CLI)")
    p.add_argument("--vad-ms", type=int, default=200, help="Silêncio mínimo em ms (CLI)")

    p.add_argument("--test", action="store_true", help="Executa testes unitários e sai")
    return p.parse_args(argv)


def coletar_entradas(args: argparse.Namespace) -> List[str]:
    arquivos: List[str] = []
    if args.inputs:
        for p in args.inputs:
            if os.path.isfile(p):
                arquivos.append(p)
            else:
                print(f"⚠️  Ignorando (não existe): {p}")
    if args.glob:
        arquivos.extend(sorted(glob.glob(args.glob)))
    vistos, unicos = set(), []
    for a in arquivos:
        if a not in vistos:
            unicos.append(a); vistos.add(a)
    return unicos


def run_cli(args: argparse.Namespace) -> int:
    entradas = coletar_entradas(args)
    if not entradas:
        print("ℹ️  Nenhum arquivo informado. Use -i arquivo.wav [outros] ou -g 'pasta/*.mp3'")
        return 1

    try:
        print(f"🧠 Carregando modelo {args.model} (cuda={args.cuda}, compute={args.compute})...")
        model = carregar_modelo(args.model, args.cuda, args.compute)
    except Exception as e:
        print("❌ Erro ao carregar o modelo:", e, file=sys.stderr)
        return 2

    os.makedirs(args.outdir, exist_ok=True)

    total_ok = 0
    for path in entradas:
        base = os.path.splitext(os.path.basename(path))[0]
        base_out = os.path.join(args.outdir, base)
        print(f"\n🎧 Arquivo: {path}")

        def on_seg(s, e, t):
            print(f"[{s:.2f}s - {e:.2f}s] {t}")

        try:
            texto_limpo, texto_tempo, linhas_json = transcrever_arquivo(
                model, path, args.vad, args.vad_ms, on_segment=on_seg
            )
        except Exception as e:
            print(f"❌ Falha na transcrição de '{path}': {e}", file=sys.stderr)
            continue

        try:
            salvar_saidas(base_out, texto_limpo, texto_tempo, linhas_json)
        except Exception as e:
            print(f"❌ Falha ao salvar saídas de '{path}': {e}", file=sys.stderr)
            continue

        print(f"✅ Salvo: {base_out}_transcricao.txt | {base_out}_com_timestamps.txt | {base_out}_transcricao.json")
        total_ok += 1

    print(f"\nResumo: {total_ok}/{len(entradas)} arquivo(s) processados.")
    return 0 if total_ok == len(entradas) else 3


# ========================= Testes =========================

def run_tests() -> int:
    import unittest

    class SegmentFake:
        def __init__(self, start, end, text):
            self.start, self.end, self.text = start, end, text

    class InfoFake:
        def __init__(self):
            self.duration = 10.0
            self.language = "pt"

    class ModelFake:
        def transcribe(self, path, vad_filter=True, vad_parameters=None):
            segs = [SegmentFake(0.00, 2.50, "olá mundo"), SegmentFake(2.50, 4.00, "isto é um teste")]
            return segs, InfoFake()

    class T(unittest.TestCase):
        def test_transcrever_arquivo(self):
            texto_limpo, texto_tempo, linhas_json = transcrever_arquivo(ModelFake(), "fake.wav", True, 200)
            self.assertEqual(texto_limpo, "olá mundo isto é um teste")
            self.assertIn("[0.00s - 2.50s] olá mundo", texto_tempo)
            self.assertEqual(len(linhas_json), 2)
        def test_salvar_saidas(self):
            import tempfile, os, json
            with tempfile.TemporaryDirectory() as tmp:
                base = os.path.join(tmp, "saida")
                salvar_saidas(base, "texto limpo", "[0-1] a", [{"inicio":0,"fim":1,"texto":"a"}], True, True, True)
                self.assertTrue(os.path.exists(base+"_transcricao.txt"))
                self.assertTrue(os.path.exists(base+"_com_timestamps.txt"))
                self.assertTrue(os.path.exists(base+"_transcricao.json"))
                with open(base+"_transcricao.json","r",encoding="utf-8") as f:
                    data=json.load(f)
                    self.assertEqual(data[0]["texto"],"a")

    ok = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(T)).wasSuccessful()
    return 0 if ok else 1


# ========================= Entrypoint =========================
if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    if args.test:
        sys.exit(run_tests())
    if args.cli:
        sys.exit(run_cli(args))
    # Default: GUI
    try:
        run_gui()
        sys.exit(0)
    except Exception as e:
        print("⚠️  GUI indisponível, caindo para CLI. Motivo:", e)
        # Fallback CLI básico sem entradas → mensagem útil
        if not (args.inputs or args.glob):
            print("Use --cli -i arquivo.wav [outros] ou --cli -g 'pasta/*.mp3'")
            sys.exit(1)
        sys.exit(run_cli(args))
