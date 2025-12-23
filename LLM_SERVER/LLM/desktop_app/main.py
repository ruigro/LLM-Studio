from __future__ import annotations

import sys
from pathlib import Path

from PySide6.QtCore import Qt, QProcess
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QTabWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QPushButton, QFileDialog, QComboBox, QTextEdit, QPlainTextEdit,
    QSpinBox, QDoubleSpinBox, QMessageBox, QListWidget, QListWidgetItem, QSplitter
)

from core.models import DEFAULT_BASE_MODELS, search_hf_models, download_hf_model, list_local_adapters, list_local_downloads, get_app_root
from core.training import TrainingConfig, default_output_dir, build_finetune_cmd
from core.inference import InferenceConfig, build_run_adapter_cmd


APP_TITLE = "LLM Fine-tuning Studio (Desktop - No Streamlit)"


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(APP_TITLE)
        self.resize(1200, 800)

        self.root = get_app_root()

        tabs = QTabWidget()
        tabs.addTab(self._build_models_tab(), "Models")
        tabs.addTab(self._build_train_tab(), "Train")
        tabs.addTab(self._build_test_tab(), "Test")
        tabs.addTab(self._build_logs_tab(), "Logs")

        self.setCentralWidget(tabs)

        self.train_proc: QProcess | None = None
        self._refresh_locals()

    # ---------------- Models tab ----------------
    def _build_models_tab(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)

        search_row = QHBoxLayout()
        self.hf_query = QLineEdit()
        self.hf_query.setPlaceholderText("Hugging Face search (e.g., Qwen2.5 bnb 4bit)")
        self.hf_search_btn = QPushButton("Search")
        self.hf_search_btn.clicked.connect(self._hf_search)
        search_row.addWidget(self.hf_query)
        search_row.addWidget(self.hf_search_btn)
        layout.addLayout(search_row)

        layout.addWidget(QLabel("Search results:"))
        self.hf_results = QListWidget()
        layout.addWidget(self.hf_results, 2)

        dl_row = QHBoxLayout()
        self.hf_target_dir = QLineEdit(str(self.root / "hf_models"))
        self.hf_browse_btn = QPushButton("Browse…")
        self.hf_browse_btn.clicked.connect(self._browse_hf_target)
        self.hf_download_btn = QPushButton("Download selected")
        self.hf_download_btn.clicked.connect(self._hf_download_selected)
        dl_row.addWidget(QLabel("Download folder:"))
        dl_row.addWidget(self.hf_target_dir, 2)
        dl_row.addWidget(self.hf_browse_btn)
        dl_row.addWidget(self.hf_download_btn)
        layout.addLayout(dl_row)

        split = QSplitter(Qt.Horizontal)

        left = QWidget(); l = QVBoxLayout(left)
        l.addWidget(QLabel("Local downloads (project):"))
        self.local_downloads = QListWidget()
        l.addWidget(self.local_downloads, 1)
        l.addWidget(QLabel("Local adapters (fine_tuned_adapter):"))
        self.local_adapters = QListWidget()
        l.addWidget(self.local_adapters, 1)

        right = QWidget(); r = QVBoxLayout(right)
        r.addWidget(QLabel("Status:"))
        self.models_status = QPlainTextEdit()
        self.models_status.setReadOnly(True)
        self.models_status.setMaximumBlockCount(5000)
        r.addWidget(self.models_status, 1)

        split.addWidget(left)
        split.addWidget(right)
        split.setSizes([500, 700])
        layout.addWidget(split, 2)

        return w

    def _browse_hf_target(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "Select download folder", str(self.root))
        if d:
            self.hf_target_dir.setText(d)

    def _hf_search(self) -> None:
        q = self.hf_query.text().strip()
        self.hf_results.clear()
        if not q:
            return
        try:
            hits = search_hf_models(q, limit=30)
            for h in hits:
                item = QListWidgetItem(f"{h.model_id}  | downloads={h.downloads} likes={h.likes}")
                item.setData(Qt.UserRole, h.model_id)
                self.hf_results.addItem(item)
            self._log_models(f"Found {len(hits)} results for: {q}")
        except Exception as e:
            self._log_models(f"[ERROR] HF search failed: {e}")

    def _hf_download_selected(self) -> None:
        item = self.hf_results.currentItem()
        if not item:
            QMessageBox.warning(self, "Download", "Select a model in the results list.")
            return
        model_id = item.data(Qt.UserRole)
        target = Path(self.hf_target_dir.text().strip())
        try:
            self._log_models(f"Downloading {model_id} -> {target}")
            dest = download_hf_model(model_id, target)
            self._log_models(f"Download complete: {dest}")
            self._refresh_locals()
        except Exception as e:
            self._log_models(f"[ERROR] Download failed: {e}")

    def _log_models(self, msg: str) -> None:
        self.models_status.appendPlainText(msg)

    # ---------------- Train tab ----------------
    def _build_train_tab(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)

        row1 = QHBoxLayout()
        self.train_base_model = QComboBox()
        self.train_base_model.setEditable(True)
        self.train_base_model.addItems(DEFAULT_BASE_MODELS)
        row1.addWidget(QLabel("Base model:"))
        row1.addWidget(self.train_base_model, 3)
        layout.addLayout(row1)

        row2 = QHBoxLayout()
        self.train_data_path = QLineEdit()
        self.train_data_path.setPlaceholderText("Path to training dataset (.jsonl recommended)")
        btn = QPushButton("Browse…")
        btn.clicked.connect(self._browse_train_data)
        row2.addWidget(QLabel("Dataset:"))
        row2.addWidget(self.train_data_path, 3)
        row2.addWidget(btn)
        layout.addLayout(row2)

        row3 = QHBoxLayout()
        self.train_out_dir = QLineEdit(str(default_output_dir()))
        btn2 = QPushButton("Browse…")
        btn2.clicked.connect(self._browse_train_out)
        row3.addWidget(QLabel("Output dir:"))
        row3.addWidget(self.train_out_dir, 3)
        row3.addWidget(btn2)
        layout.addLayout(row3)

        row4 = QHBoxLayout()
        self.train_epochs = QSpinBox(); self.train_epochs.setRange(1, 1000); self.train_epochs.setValue(1)
        self.train_batch = QSpinBox(); self.train_batch.setRange(1, 512); self.train_batch.setValue(1)
        self.train_lr = QDoubleSpinBox(); self.train_lr.setDecimals(8); self.train_lr.setRange(1e-8, 1.0); self.train_lr.setValue(2e-4)
        self.train_lr.setSingleStep(1e-4)
        row4.addWidget(QLabel("Epochs:")); row4.addWidget(self.train_epochs)
        row4.addWidget(QLabel("Batch:")); row4.addWidget(self.train_batch)
        row4.addWidget(QLabel("LR:")); row4.addWidget(self.train_lr)
        row4.addStretch(1)
        layout.addLayout(row4)

        ctrl = QHBoxLayout()
        self.train_start = QPushButton("Start training")
        self.train_stop = QPushButton("Stop")
        self.train_stop.setEnabled(False)
        self.train_start.clicked.connect(self._start_training)
        self.train_stop.clicked.connect(self._stop_training)
        ctrl.addWidget(self.train_start)
        ctrl.addWidget(self.train_stop)
        ctrl.addStretch(1)
        layout.addLayout(ctrl)

        layout.addWidget(QLabel("Training output:"))
        self.train_log = QPlainTextEdit()
        self.train_log.setReadOnly(True)
        self.train_log.setMaximumBlockCount(10000)
        layout.addWidget(self.train_log, 1)

        return w

    def _browse_train_data(self) -> None:
        f, _ = QFileDialog.getOpenFileName(self, "Select dataset file", str(self.root), "Data files (*.jsonl *.json *.txt);;All files (*)")
        if f:
            self.train_data_path.setText(f)

    def _browse_train_out(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "Select output folder", str(self.root))
        if d:
            self.train_out_dir.setText(d)

    def _start_training(self) -> None:
        if self.train_proc is not None:
            QMessageBox.information(self, "Training", "Training is already running.")
            return
        data_path = self.train_data_path.text().strip()
        if not data_path:
            QMessageBox.warning(self, "Training", "Select a dataset file.")
            return

        cfg = TrainingConfig(
            base_model=self.train_base_model.currentText().strip(),
            data_path=Path(data_path),
            output_dir=Path(self.train_out_dir.text().strip()),
            epochs=int(self.train_epochs.value()),
            batch_size=int(self.train_batch.value()),
            learning_rate=float(self.train_lr.value()),
        )

        cmd = build_finetune_cmd(cfg)

        proc = QProcess(self)
        proc.setProgram(cmd[0])
        proc.setArguments(cmd[1:])
        proc.setWorkingDirectory(str(self.root))
        proc.setProcessChannelMode(QProcess.MergedChannels)
        proc.readyReadStandardOutput.connect(lambda: self._append_proc_output(proc, self.train_log))
        proc.finished.connect(self._train_finished)

        self.train_log.appendPlainText(">> " + " ".join(cmd))
        proc.start()
        if not proc.waitForStarted(5000):
            QMessageBox.critical(self, "Training", "Failed to start training process.")
            return

        self.train_proc = proc
        self.train_start.setEnabled(False)
        self.train_stop.setEnabled(True)

    def _stop_training(self) -> None:
        if self.train_proc is None:
            return
        self.train_log.appendPlainText("\n[INFO] Terminating training process...")
        self.train_proc.terminate()
        if not self.train_proc.waitForFinished(5000):
            self.train_proc.kill()

    def _train_finished(self) -> None:
        self.train_log.appendPlainText("\n[INFO] Training process finished.")
        self.train_proc = None
        self.train_start.setEnabled(True)
        self.train_stop.setEnabled(False)
        self._refresh_locals()

    # ---------------- Test tab ----------------
    def _build_test_tab(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)

        top = QHBoxLayout()
        self.test_mode = QComboBox()
        self.test_mode.addItems(["Base model only", "Adapter (fine-tuned)"])
        top.addWidget(QLabel("Mode:"))
        top.addWidget(self.test_mode)
        top.addStretch(1)
        layout.addLayout(top)

        sel = QHBoxLayout()
        self.test_base_model = QComboBox()
        self.test_base_model.setEditable(True)
        self.test_base_model.addItems(DEFAULT_BASE_MODELS)
        self.test_adapter = QComboBox()
        sel.addWidget(QLabel("Base model:"))
        sel.addWidget(self.test_base_model, 2)
        sel.addWidget(QLabel("Adapter:"))
        sel.addWidget(self.test_adapter, 2)
        layout.addLayout(sel)

        split = QSplitter(Qt.Horizontal)
        left = QWidget(); llay = QVBoxLayout(left)
        right = QWidget(); rlay = QVBoxLayout(right)

        self.test_prompt = QTextEdit()
        self.test_prompt.setPlaceholderText("Write a prompt…")
        self.test_run = QPushButton("Run")
        self.test_run.clicked.connect(self._run_test)

        llay.addWidget(QLabel("Prompt:"))
        llay.addWidget(self.test_prompt, 1)
        llay.addWidget(self.test_run)

        self.test_output = QPlainTextEdit()
        self.test_output.setReadOnly(True)
        self.test_output.setMaximumBlockCount(10000)
        rlay.addWidget(QLabel("Output:"))
        rlay.addWidget(self.test_output, 1)

        split.addWidget(left)
        split.addWidget(right)
        split.setSizes([500, 700])

        layout.addWidget(split, 1)
        return w

    def _run_test(self) -> None:
        prompt = self.test_prompt.toPlainText().strip()
        if not prompt:
            return

        mode = self.test_mode.currentText()
        base_model = self.test_base_model.currentText().strip()
        adapter = self.test_adapter.currentText().strip()

        cfg = InferenceConfig(prompt=prompt, base_model=base_model)

        if mode == "Adapter (fine-tuned)":
            if not adapter:
                QMessageBox.warning(self, "Test", "Select an adapter.")
                return
            cfg.adapter_dir = self.root / "fine_tuned_adapter" / adapter

        cmd = build_run_adapter_cmd(cfg)

        proc = QProcess(self)
        proc.setProgram(cmd[0])
        proc.setArguments(cmd[1:])
        proc.setWorkingDirectory(str(self.root))
        proc.setProcessChannelMode(QProcess.MergedChannels)
        proc.readyReadStandardOutput.connect(lambda: self._append_proc_output(proc, self.test_output))

        self.test_output.clear()
        self.test_output.appendPlainText(">> " + " ".join(cmd))
        proc.start()

    # ---------------- Logs tab ----------------
    def _build_logs_tab(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)

        row = QHBoxLayout()
        self.logs_refresh = QPushButton("Refresh")
        self.logs_refresh.clicked.connect(self._refresh_locals)
        row.addWidget(self.logs_refresh)
        row.addStretch(1)
        layout.addLayout(row)

        split = QSplitter(Qt.Horizontal)
        self.logs_list = QListWidget()
        self.logs_list.itemSelectionChanged.connect(self._open_selected_log)
        self.logs_view = QPlainTextEdit()
        self.logs_view.setReadOnly(True)
        self.logs_view.setMaximumBlockCount(20000)

        split.addWidget(self.logs_list)
        split.addWidget(self.logs_view)
        split.setSizes([300, 900])
        layout.addWidget(split, 1)
        return w

    def _open_selected_log(self) -> None:
        item = self.logs_list.currentItem()
        if not item:
            return
        path = Path(item.data(Qt.UserRole))
        try:
            data = path.read_text(encoding="utf-8", errors="replace")
            self.logs_view.setPlainText(data[-20000:])
        except Exception as e:
            self.logs_view.setPlainText(f"[ERROR] Could not read {path}: {e}")

    # ---------------- Helpers ----------------
    def _append_proc_output(self, proc: QProcess, widget: QPlainTextEdit) -> None:
        data = proc.readAllStandardOutput().data().decode("utf-8", errors="replace")
        if data:
            widget.appendPlainText(data.rstrip("\n"))

    def _refresh_locals(self) -> None:
        # Local downloads and adapters
        self.local_downloads.clear()
        for name in list_local_downloads(self.root / "hf_models"):
            self.local_downloads.addItem(name)

        adapters = list_local_adapters(self.root / "fine_tuned_adapter")
        self.local_adapters.clear()
        for name in adapters:
            self.local_adapters.addItem(name)

        self.test_adapter.clear()
        self.test_adapter.addItems(adapters)

        # log list from repo root
        self.logs_list.clear()
        for p in sorted(self.root.glob("*training*.txt")) + sorted(self.root.glob("*log*.txt")):
            it = QListWidgetItem(str(p.name))
            it.setData(Qt.UserRole, str(p))
            self.logs_list.addItem(it)

    # these are set in models tab construction
    local_downloads: QListWidget
    local_adapters: QListWidget


def main() -> int:
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
