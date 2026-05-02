"""RVC Voice Bank Packer for OpenUtau — retro UTAU 2008 / classic IE style GUI.

Square corners, 3D sunken/raised borders, system gray, MS Gothic-esque fonts.
Trilingual: English / 日本語 / 中文
"""

import sys
import os
import tkinter as tk
from tkinter import filedialog, messagebox
from pathlib import Path
import threading

# PyInstaller puts data in _internal/, normal run uses parent dir
if getattr(sys, 'frozen', False):
    ADAPTER_ROOT = Path(sys._MEIPASS)
else:
    ADAPTER_ROOT = Path(__file__).resolve().parents[1]

# ── i18n ───────────────────────────────────────────────────────────────────

LANGS = ["en", "ja", "zh"]

_STRINGS = {
    "app_title": {
        "en": "RVC Voice Bank Packer for OpenUtau",
        "ja": "RVC ボイスバンク パッカー for OpenUtau",
        "zh": "RVC 声库打包工具 for OpenUtau",
    },
    "title_bar": {
        "en": "  RVC Voice Bank Packer  v1.1",
        "ja": "  RVC ボイスバンク パッカー  v1.1",
        "zh": "  RVC 声库打包工具  v1.1",
    },
    "sec_voice": {
        "en": " Voice Info ",
        "ja": " ボイス情報 ",
        "zh": " 声库信息 ",
    },
    "voice_name": {
        "en": "Voice Name:",
        "ja": "ボイス名:",
        "zh": "声库名称:",
    },
    "author": {
        "en": "Author:",
        "ja": "作者:",
        "zh": "作者:",
    },
    "language": {
        "en": "Language:",
        "ja": "言語:",
        "zh": "语言:",
    },
    "avatar": {
        "en": "Avatar (.png):",
        "ja": "アバター (.png):",
        "zh": "头像 (.png):",
    },
    "sec_rvc": {
        "en": " RVC Model ",
        "ja": " RVC モデル ",
        "zh": " RVC 模型 ",
    },
    "rvc_pth": {
        "en": "RVC .pth:",
        "ja": "RVC .pth:",
        "zh": "RVC .pth:",
    },
    "index_file": {
        "en": "Index .index:",
        "ja": "インデックス:",
        "zh": "索引文件:",
    },
    "index_rate": {
        "en": "Index Rate:",
        "ja": "インデックスレート:",
        "zh": "索引混合率:",
    },
    "sec_output": {
        "en": " Output ",
        "ja": " 出力先 ",
        "zh": " 输出 ",
    },
    "output_dir": {
        "en": "Output Dir:",
        "ja": "出力フォルダ:",
        "zh": "输出目录:",
    },
    "btn_export": {
        "en": "   Export Voice Bank   ",
        "ja": "   ボイスバンクを出力   ",
        "zh": "   导出声库   ",
    },
    "btn_exporting": {
        "en": "Exporting...",
        "ja": "出力中...",
        "zh": "导出中...",
    },
    "browse": {
        "en": "Browse...",
        "ja": "参照...",
        "zh": "浏览...",
    },
    "log": {
        "en": " Log ",
        "ja": " ログ ",
        "zh": " 日志 ",
    },
    "ready": {
        "en": "Ready.",
        "ja": "準備完了",
        "zh": "就绪",
    },
    "none": {
        "en": "(none)",
        "ja": "(なし)",
        "zh": "(无)",
    },
    "ui_lang": {
        "en": "UI Language:",
        "ja": "表示言語:",
        "zh": "界面语言:",
    },
    "success": {
        "en": "Voice bank exported successfully!",
        "ja": "ボイスバンクの出力が完了しました！",
        "zh": "声库导出成功！",
    },
    "err_name": {
        "en": "Voice name is required.",
        "ja": "ボイス名を入力してください。",
        "zh": "请输入声库名称。",
    },
    "err_rvc": {
        "en": "Please select a valid RVC .pth file.",
        "ja": "有効な RVC .pth ファイルを選択してください。",
        "zh": "请选择有效的 RVC .pth 文件。",
    },
    "err_output": {
        "en": "Please select an output directory.",
        "ja": "出力フォルダを選択してください。",
        "zh": "请选择输出目录。",
    },
    "dlg_rvc": {
        "en": "Select RVC Model (.pth)",
        "ja": "RVC モデルを選択 (.pth)",
        "zh": "选择 RVC 模型 (.pth)",
    },
    "dlg_index": {
        "en": "Select Index File",
        "ja": "インデックスファイルを選択",
        "zh": "选择索引文件",
    },
    "dlg_avatar": {
        "en": "Select Avatar Image",
        "ja": "アバター画像を選択",
        "zh": "选择头像图片",
    },
    "dlg_output": {
        "en": "Select Output Directory",
        "ja": "出力フォルダを選択",
        "zh": "选择输出目录",
    },
}

# ── Color scheme: teal/steel-blue (matched to feather.ico #5f7b86) ────────
BG          = "#e4ecef"
BG_LIGHT    = "#f0f5f7"
FG          = "#1e3640"
FG_DIM      = "#5f7b86"
ENTRY_BG    = "#ffffff"
BTN_BG      = "#c8d5da"
BTN_ACTIVE  = "#b0c2c9"
ACCENT      = "#3a5561"
STATUS_BG   = "#e4ecef"
TITLE_BG    = "#3a5561"
TITLE_FG    = "#ffffff"

FONT_LABEL   = ("MS Gothic", 9)
FONT_ENTRY   = ("MS Gothic", 9)
FONT_BUTTON  = ("MS Gothic", 9, "bold")
FONT_TITLE   = ("MS Gothic", 11, "bold")
FONT_STATUS  = ("MS Gothic", 8)
FONT_LOG     = ("Consolas", 8)  # Consolas shows \ correctly, not ¥


class RetroFrame(tk.Frame):
    def __init__(self, master, **kw):
        kw.setdefault("bg", BG)
        kw.setdefault("relief", "raised")
        kw.setdefault("bd", 2)
        super().__init__(master, **kw)


class RetroLabel(tk.Label):
    def __init__(self, master, **kw):
        kw.setdefault("bg", BG)
        kw.setdefault("fg", FG)
        kw.setdefault("font", FONT_LABEL)
        kw.setdefault("anchor", "w")
        super().__init__(master, **kw)


class RetroEntry(tk.Entry):
    def __init__(self, master, **kw):
        kw.setdefault("bg", ENTRY_BG)
        kw.setdefault("fg", FG)
        kw.setdefault("font", FONT_ENTRY)
        kw.setdefault("relief", "sunken")
        kw.setdefault("bd", 2)
        kw.setdefault("insertbackground", FG)
        super().__init__(master, **kw)


class RetroButton(tk.Button):
    def __init__(self, master, **kw):
        kw.setdefault("bg", BTN_BG)
        kw.setdefault("fg", FG)
        kw.setdefault("font", FONT_BUTTON)
        kw.setdefault("relief", "raised")
        kw.setdefault("bd", 2)
        kw.setdefault("activebackground", BTN_ACTIVE)
        kw.setdefault("cursor", "hand2")
        kw.setdefault("padx", 8)
        kw.setdefault("pady", 2)
        super().__init__(master, **kw)


class RetroScale(tk.Scale):
    def __init__(self, master, **kw):
        kw.setdefault("bg", BG)
        kw.setdefault("fg", FG)
        kw.setdefault("font", FONT_LABEL)
        kw.setdefault("troughcolor", BG_LIGHT)
        kw.setdefault("relief", "sunken")
        kw.setdefault("bd", 2)
        kw.setdefault("highlightthickness", 0)
        kw.setdefault("sliderrelief", "raised")
        super().__init__(master, **kw)


def _s(key, lang):
    return _STRINGS.get(key, {}).get(lang, _STRINGS.get(key, {}).get("en", key))


class PackerGUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.lang = "en"
        self.root.configure(bg=BG)
        self.root.resizable(False, False)
        self.root.geometry("560x680")

        # set icon (try .ico for window, fallback to .png)
        for icon_name in ("feather.ico", "icon.png"):
            icon_path = ADAPTER_ROOT / "assets" / icon_name
            if icon_path.exists():
                try:
                    if icon_name.endswith(".ico"):
                        self.root.iconbitmap(str(icon_path))
                    else:
                        self._icon = tk.PhotoImage(file=str(icon_path))
                        self.root.iconphoto(True, self._icon)
                    break
                except Exception:
                    continue

        self.var_name = tk.StringVar(value="")
        self.var_rvc_path = tk.StringVar(value="")
        self.var_index_path = tk.StringVar(value=_s("none", self.lang))
        self.var_index_rate = tk.DoubleVar(value=0.75)
        self.var_avatar_path = tk.StringVar(value=_s("none", self.lang))
        self.var_lang = tk.StringVar(value="ja")
        self.var_author = tk.StringVar(value="")
        self.var_output = tk.StringVar(value="")
        self.var_ui_lang = tk.StringVar(value="en")

        # keep refs to all translatable widgets
        self._tw = {}
        self._build_ui()
        self._apply_lang("en")

    def _build_ui(self):
        # ── Title bar ──
        title_frame = tk.Frame(self.root, bg=TITLE_BG, height=24)
        title_frame.pack(fill="x", padx=2, pady=(2, 0))
        title_frame.pack_propagate(False)
        self._tw["title_bar"] = tk.Label(
            title_frame, text="", bg=TITLE_BG, fg=TITLE_FG,
            font=("MS Gothic", 9, "bold"), anchor="w")
        self._tw["title_bar"].pack(fill="x", padx=4, pady=2)

        # ── Main ──
        main = RetroFrame(self.root, bd=2, relief="sunken")
        main.pack(fill="both", expand=True, padx=4, pady=4)

        # ── UI Language selector (top right) ──
        lang_row = tk.Frame(main, bg=BG)
        lang_row.pack(fill="x", padx=6, pady=(4, 0))
        self._tw["ui_lang"] = RetroLabel(lang_row, text="", width=14)
        self._tw["ui_lang"].pack(side="left")
        for code, label in [("en", "English"), ("ja", "日本語"), ("zh", "中文")]:
            tk.Radiobutton(
                lang_row, text=label, variable=self.var_ui_lang, value=code,
                bg=BG, fg=FG, font=FONT_LABEL, activebackground=BG,
                selectcolor=ENTRY_BG, indicatoron=True, relief="flat",
                command=lambda c=code: self._apply_lang(c),
            ).pack(side="left", padx=(0, 8))

        # ── Section: Voice Info ──
        sec1 = RetroFrame(main, bd=1)
        sec1.pack(fill="x", padx=6, pady=(4, 4))
        self._tw["sec_voice"] = tk.Label(
            sec1, text="", bg=BG, fg=ACCENT, font=FONT_TITLE)
        self._tw["sec_voice"].pack(anchor="w", padx=4, pady=(4, 2))

        row = tk.Frame(sec1, bg=BG)
        row.pack(fill="x", padx=8, pady=2)
        self._tw["voice_name"] = RetroLabel(row, text="", width=14)
        self._tw["voice_name"].pack(side="left")
        RetroEntry(row, textvariable=self.var_name, width=30).pack(
            side="left", fill="x", expand=True, padx=(4, 8))

        row = tk.Frame(sec1, bg=BG)
        row.pack(fill="x", padx=8, pady=2)
        self._tw["author"] = RetroLabel(row, text="", width=14)
        self._tw["author"].pack(side="left")
        RetroEntry(row, textvariable=self.var_author, width=30).pack(
            side="left", fill="x", expand=True, padx=(4, 8))

        row = tk.Frame(sec1, bg=BG)
        row.pack(fill="x", padx=8, pady=2)
        self._tw["language"] = RetroLabel(row, text="", width=14)
        self._tw["language"].pack(side="left")
        lf = tk.Frame(row, bg=BG)
        lf.pack(side="left")
        for val, txt in [("ja", "日本語"), ("zh", "中文")]:
            tk.Radiobutton(
                lf, text=txt, variable=self.var_lang, value=val,
                bg=BG, fg=FG, font=FONT_LABEL, activebackground=BG,
                selectcolor=ENTRY_BG, indicatoron=True, relief="flat",
            ).pack(side="left", padx=(0, 12))

        row = tk.Frame(sec1, bg=BG)
        row.pack(fill="x", padx=8, pady=(2, 6))
        self._tw["avatar"] = RetroLabel(row, text="", width=14)
        self._tw["avatar"].pack(side="left")
        RetroEntry(row, textvariable=self.var_avatar_path, width=22,
                   state="readonly").pack(side="left", fill="x", expand=True, padx=(4, 4))
        self._tw["btn_avatar"] = RetroButton(row, text="", command=self._browse_avatar)
        self._tw["btn_avatar"].pack(side="left", padx=(0, 8))

        # ── Section: RVC Model ──
        sec2 = RetroFrame(main, bd=1)
        sec2.pack(fill="x", padx=6, pady=4)
        self._tw["sec_rvc"] = tk.Label(
            sec2, text="", bg=BG, fg=ACCENT, font=FONT_TITLE)
        self._tw["sec_rvc"].pack(anchor="w", padx=4, pady=(4, 2))

        row = tk.Frame(sec2, bg=BG)
        row.pack(fill="x", padx=8, pady=2)
        self._tw["rvc_pth"] = RetroLabel(row, text="", width=14)
        self._tw["rvc_pth"].pack(side="left")
        RetroEntry(row, textvariable=self.var_rvc_path, width=22,
                   state="readonly").pack(side="left", fill="x", expand=True, padx=(4, 4))
        self._tw["btn_rvc"] = RetroButton(row, text="", command=self._browse_rvc)
        self._tw["btn_rvc"].pack(side="left", padx=(0, 8))

        row = tk.Frame(sec2, bg=BG)
        row.pack(fill="x", padx=8, pady=2)
        self._tw["index_file"] = RetroLabel(row, text="", width=14)
        self._tw["index_file"].pack(side="left")
        RetroEntry(row, textvariable=self.var_index_path, width=22,
                   state="readonly").pack(side="left", fill="x", expand=True, padx=(4, 4))
        self._tw["btn_index"] = RetroButton(row, text="", command=self._browse_index)
        self._tw["btn_index"].pack(side="left", padx=(0, 8))

        row = tk.Frame(sec2, bg=BG)
        row.pack(fill="x", padx=8, pady=(2, 6))
        self._tw["index_rate"] = RetroLabel(row, text="", width=14)
        self._tw["index_rate"].pack(side="left")
        RetroScale(row, variable=self.var_index_rate, from_=0.0, to=1.0,
                   resolution=0.05, orient="horizontal", length=200,
                   showvalue=True).pack(side="left", padx=(4, 8))

        # ── Section: Output ──
        sec3 = RetroFrame(main, bd=1)
        sec3.pack(fill="x", padx=6, pady=4)
        self._tw["sec_output"] = tk.Label(
            sec3, text="", bg=BG, fg=ACCENT, font=FONT_TITLE)
        self._tw["sec_output"].pack(anchor="w", padx=4, pady=(4, 2))

        row = tk.Frame(sec3, bg=BG)
        row.pack(fill="x", padx=8, pady=(2, 6))
        self._tw["output_dir"] = RetroLabel(row, text="", width=14)
        self._tw["output_dir"].pack(side="left")
        RetroEntry(row, textvariable=self.var_output, width=22,
                   state="readonly").pack(side="left", fill="x", expand=True, padx=(4, 4))
        self._tw["btn_output"] = RetroButton(row, text="", command=self._browse_output)
        self._tw["btn_output"].pack(side="left", padx=(0, 8))

        # ── Export Button ──
        btn_frame = tk.Frame(main, bg=BG)
        btn_frame.pack(fill="x", padx=6, pady=8)
        self.btn_export = RetroButton(
            btn_frame, text="", font=("MS Gothic", 11, "bold"),
            command=self._export, bd=3)
        self.btn_export.pack(pady=4)
        self._tw["btn_export"] = self.btn_export

        # ── Log ──
        log_frame = RetroFrame(main, bd=1, relief="sunken")
        log_frame.pack(fill="both", expand=True, padx=6, pady=(0, 6))
        self._tw["log"] = tk.Label(
            log_frame, text="", bg=BG, fg=FG_DIM, font=FONT_STATUS)
        self._tw["log"].pack(anchor="w", padx=4, pady=(2, 0))
        self.log_text = tk.Text(
            log_frame, height=8, bg=ENTRY_BG, fg=FG, font=FONT_LOG,
            relief="sunken", bd=2, state="disabled", wrap="word")
        self.log_text.pack(fill="both", expand=True, padx=4, pady=(0, 4))

        # ── Status bar ──
        status = tk.Frame(self.root, bg=STATUS_BG, relief="sunken", bd=1, height=20)
        status.pack(fill="x", side="bottom", padx=2, pady=(0, 2))
        status.pack_propagate(False)
        self.status_label = tk.Label(
            status, text="", bg=STATUS_BG, fg=FG_DIM, font=FONT_STATUS, anchor="w")
        self.status_label.pack(fill="x", padx=4)

    # ── i18n ──

    def _apply_lang(self, lang_code):
        self.lang = lang_code
        self.root.title(_s("app_title", lang_code))
        for key, widget in self._tw.items():
            if key.startswith("btn_") and key not in ("btn_export",):
                widget.configure(text=_s("browse", lang_code))
            elif key == "btn_export":
                widget.configure(text=_s("btn_export", lang_code))
            else:
                widget.configure(text=_s(key, lang_code))
        self.status_label.configure(text=_s("ready", lang_code))
        none_str = _s("none", lang_code)
        if not self.var_index_path.get() or self.var_index_path.get().startswith("("):
            self.var_index_path.set(none_str)
        if not self.var_avatar_path.get() or self.var_avatar_path.get().startswith("("):
            self.var_avatar_path.set(none_str)

    # ── Browse callbacks ──

    def _browse_rvc(self):
        path = filedialog.askopenfilename(
            title=_s("dlg_rvc", self.lang),
            filetypes=[("PyTorch Model", "*.pth"), ("All Files", "*.*")])
        if path:
            self.var_rvc_path.set(path)
            self._log(f"RVC model: {self._display_path(path)}")
            if not self.var_name.get():
                self.var_name.set(Path(path).stem)

    def _browse_index(self):
        path = filedialog.askopenfilename(
            title=_s("dlg_index", self.lang),
            filetypes=[
                ("Faiss Index", "*.index"),
                ("Numpy Array", "*.npy"),
                ("All Files", "*.*"),
            ])
        if path:
            self.var_index_path.set(path)
            self._log(f"Index: {self._display_path(path)}")

    def _browse_avatar(self):
        path = filedialog.askopenfilename(
            title=_s("dlg_avatar", self.lang),
            filetypes=[("PNG Image", "*.png"), ("JPEG Image", "*.jpg;*.jpeg"),
                       ("All Files", "*.*")])
        if path:
            self.var_avatar_path.set(path)

    def _browse_output(self):
        path = filedialog.askdirectory(title=_s("dlg_output", self.lang))
        if path:
            self.var_output.set(path)

    # ── Helpers ──

    @staticmethod
    def _display_path(p):
        """Forward slashes for display (avoids MS Gothic ¥ rendering of \\)."""
        return str(p).replace("\\", "/")

    def _log(self, msg: str):
        self.log_text.configure(state="normal")
        self.log_text.insert("end", msg.replace("\\", "/") + "\n")
        self.log_text.see("end")
        self.log_text.configure(state="disabled")

    def _set_status(self, msg: str):
        self.status_label.configure(text=msg)

    def _log_safe(self, msg):
        self.root.after(0, lambda: self._log(msg))

    # ── Export ──

    def _validate(self) -> bool:
        if not self.var_name.get().strip():
            messagebox.showerror("Error", _s("err_name", self.lang))
            return False
        rvc = self.var_rvc_path.get()
        if not rvc or not Path(rvc).exists():
            messagebox.showerror("Error", _s("err_rvc", self.lang))
            return False
        if not self.var_output.get():
            messagebox.showerror("Error", _s("err_output", self.lang))
            return False
        return True

    def _export(self):
        if not self._validate():
            return
        self.btn_export.configure(state="disabled", text=_s("btn_exporting", self.lang))
        self._set_status(_s("btn_exporting", self.lang))
        self._log("=" * 50)

        def _run():
            try:
                self._do_export()
                self.root.after(0, lambda: self._set_status(
                    _s("ready", self.lang)))
                self.root.after(0, lambda: messagebox.showinfo(
                    "OK", _s("success", self.lang)))
            except Exception as e:
                self.root.after(0, lambda: self._log(f"ERROR: {e}"))
                self.root.after(0, lambda: self._set_status(f"Error: {e}"))
                self.root.after(0, lambda: messagebox.showerror("Error", str(e)))
            finally:
                self.root.after(0, lambda: self.btn_export.configure(
                    state="normal", text=_s("btn_export", self.lang)))

        threading.Thread(target=_run, daemon=True).start()

    def _do_export(self):
        name = self.var_name.get().strip()
        rvc_path = self.var_rvc_path.get()
        index_path = self.var_index_path.get()
        index_rate = self.var_index_rate.get()
        lang = self.var_lang.get()
        output_base = Path(self.var_output.get())
        author = self.var_author.get().strip()
        avatar = self.var_avatar_path.get()
        none_str = _s("none", self.lang)

        safe_name = "".join(c for c in name if c.isalnum() or c in " _-").strip()
        output_dir = output_base / safe_name

        # resolve index: support both .index (faiss) and .npy
        idx_npy = None
        if index_path and index_path != none_str and Path(index_path).exists():
            p = Path(index_path)
            if p.suffix == ".npy":
                idx_npy = str(p)
                self._log_safe(f"Index (.npy): {self._display_path(p)}")
            elif p.suffix == ".index":
                self._log_safe(f"Index (.index): {self._display_path(p)}")
                try:
                    import faiss
                    idx = faiss.read_index(str(p))
                    import numpy as np
                    big_npy = idx.reconstruct_n(0, idx.ntotal)
                    tmp_npy = ADAPTER_ROOT / "output" / f"{safe_name}_index.npy"
                    tmp_npy.parent.mkdir(parents=True, exist_ok=True)
                    np.save(str(tmp_npy), big_npy)
                    idx_npy = str(tmp_npy)
                    self._log_safe(f"  Converted: {idx.ntotal} vectors extracted")
                except ImportError:
                    self._log_safe("  WARNING: faiss not available, index skipped")
                except Exception as e:
                    self._log_safe(f"  WARNING: failed to read .index: {e}")

        avatar_file = ""
        if avatar and avatar != none_str and Path(avatar).exists():
            avatar_file = avatar

        try:
            from src.pack_voicebank import pack_voicebank
        except ImportError:
            from pack_voicebank import pack_voicebank

        pack_voicebank(
            rvc_pth_path=rvc_path,
            output_dir=str(output_dir),
            voice_name=name,
            language=lang,
            index_npy_path=idx_npy,
            index_rate=index_rate,
            author=author,
            avatar_path=avatar_file,
            on_progress=self._log_safe,
        )


def main():
    root = tk.Tk()
    try:
        root.tk.call("ttk::style", "theme", "use", "winnative")
    except tk.TclError:
        pass
    app = PackerGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
