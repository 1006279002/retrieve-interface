from __future__ import annotations

import random
import traceback
from pathlib import Path
from typing import Optional

from PyQt5.QtCore import QSize, Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont, QFontDatabase, QIcon, QPixmap
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QComboBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QPlainTextEdit,
    QSizePolicy,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from .backend import FusionDatasetSession, FusionWorkspace, QuerySample, RetrievalResult


class DatasetLoaderThread(QThread):
    loaded = pyqtSignal(object)
    failed = pyqtSignal(str)
    status = pyqtSignal(str)

    def __init__(self, workspace: FusionWorkspace, dataset_key: str) -> None:
        super().__init__()
        self.workspace = workspace
        self.dataset_key = dataset_key

    def _emit_status(self, message: str, current: Optional[int] = None, total: Optional[int] = None) -> None:
        if current is not None and total:
            self.status.emit("{} ({}/{})".format(message, current, total))
            return
        self.status.emit(message)

    def run(self) -> None:
        try:
            session = self.workspace.get_session(self.dataset_key)
            session.ensure_ready(status_callback=self._emit_status)
            self.loaded.emit(session)
        except Exception:
            self.failed.emit(traceback.format_exc())


class RetrievalThread(QThread):
    completed = pyqtSignal(object)
    failed = pyqtSignal(str)
    status = pyqtSignal(str)

    def __init__(self, session: FusionDatasetSession, sample: QuerySample, sketch_path: Path, text: str, top_k: int) -> None:
        super().__init__()
        self.session = session
        self.sample = sample
        self.sketch_path = sketch_path
        self.text = text
        self.top_k = top_k

    def run(self) -> None:
        try:
            self.status.emit("Retrieving top-{} images".format(self.top_k))
            result = self.session.retrieve(
                sample=self.sample,
                sketch_path=self.sketch_path,
                text=self.text,
                top_k=self.top_k,
            )
            self.completed.emit(result)
        except Exception:
            self.failed.emit(traceback.format_exc())


class StableImageLabel(QLabel):
    def __init__(self, preferred_size: QSize, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._preferred_size = preferred_size

    def sizeHint(self) -> QSize:
        return QSize(self._preferred_size)

    def minimumSizeHint(self) -> QSize:
        return QSize(
            max(96, int(self._preferred_size.width() * 0.55)),
            max(108, int(self._preferred_size.height() * 0.55)),
        )


class ImageCard(QWidget):
    def __init__(self, title: str, image_size: QSize, accent: bool = False, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._image_size = image_size
        self._accent = accent
        self._pixmap: Optional[QPixmap] = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        self.title_label = QLabel(title)
        self.title_label.setObjectName("cardTitle")
        self.title_label.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        layout.addWidget(self.title_label)

        self.image_label = StableImageLabel(self._image_size)
        self.image_label.setText("No image")
        self.image_label.setObjectName("imageFrameAccent" if accent else "imageFrame")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout.addWidget(self.image_label, alignment=Qt.AlignCenter)

        self.caption_label = QLabel("")
        self.caption_label.setObjectName("captionText")
        self.caption_label.setWordWrap(True)
        self.caption_label.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        layout.addWidget(self.caption_label)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    def _render_pixmap(self) -> None:
        if self._pixmap is None:
            return
        target_size = self.image_label.size()
        if target_size.width() <= 0 or target_size.height() <= 0:
            return
        scaled = self._pixmap.scaled(
            target_size,
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        self.image_label.setPixmap(scaled)

    def clear(self, title: Optional[str] = None) -> None:
        if title is not None:
            self.title_label.setText(title)
        self._pixmap = None
        self.image_label.clear()
        self.image_label.setText("No image")
        self.caption_label.setText("")

    def set_content(self, title: str, image_path: Optional[Path], caption: str = "") -> None:
        self.title_label.setText(title)
        self.caption_label.setText(caption)
        if image_path is None or not image_path.exists():
            self._pixmap = None
            self.image_label.clear()
            self.image_label.setText("Missing image")
            return

        pixmap = QPixmap(str(image_path))
        if pixmap.isNull():
            self._pixmap = None
            self.image_label.clear()
            self.image_label.setText("Invalid image")
            return

        self._pixmap = pixmap
        self._render_pixmap()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._render_pixmap()


class MainWindow(QMainWindow):
    def __init__(self, workspace: Optional[FusionWorkspace] = None) -> None:
        super().__init__()
        self.workspace = workspace or FusionWorkspace()
        self.current_session: Optional[FusionDatasetSession] = None
        self._load_thread: Optional[DatasetLoaderThread] = None
        self._retrieval_thread: Optional[RetrievalThread] = None

        self.setWindowTitle("STFusionIR Retrieval Desktop")
        self.setMinimumSize(1320, 840)

        self._build_ui()
        self._configure_initial_geometry()
        self._apply_styles()
        self._populate_datasets()
        self._start_loading_current_dataset()

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)

        root_layout = QVBoxLayout(central)
        root_layout.setContentsMargins(20, 18, 20, 18)
        root_layout.setSpacing(14)

        header_layout = QHBoxLayout()
        header_layout.setSpacing(10)
        root_layout.addLayout(header_layout)

        header_text = QVBoxLayout()
        header_text.setSpacing(4)
        header_layout.addLayout(header_text, stretch=1)

        title_label = QLabel("Sketch + Text Retrieval")
        title_font = QFont()
        title_font.setPointSize(21)
        title_font.setBold(True)
        title_label.setFont(title_font)
        header_text.addWidget(title_label)

        subtitle = QLabel("Browse a query sample, choose one sketch candidate, adjust the prompt, then retrieve the top-5 images.")
        subtitle.setObjectName("subtitle")
        subtitle.setWordWrap(True)
        header_text.addWidget(subtitle)

        self.header_status = QLabel("Preparing workspace")
        self.header_status.setObjectName("headerPill")
        header_layout.addWidget(self.header_status, alignment=Qt.AlignTop)

        self.main_splitter = QSplitter(Qt.Horizontal)
        self.main_splitter.setHandleWidth(8)
        root_layout.addWidget(self.main_splitter, stretch=1)

        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(14)
        self.main_splitter.addWidget(left_panel)

        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(14)
        self.main_splitter.addWidget(right_panel)
        self.main_splitter.setSizes([380, 940])

        dataset_group = QGroupBox("Dataset")
        dataset_layout = QVBoxLayout(dataset_group)
        dataset_layout.setSpacing(10)
        left_layout.addWidget(dataset_group)

        self.dataset_combo = QComboBox()
        self.dataset_combo.currentIndexChanged.connect(self._start_loading_current_dataset)
        dataset_layout.addWidget(self.dataset_combo)

        self.dataset_info_label = QLabel("Loading dataset summary...")
        self.dataset_info_label.setWordWrap(True)
        self.dataset_info_label.setObjectName("mutedText")
        dataset_layout.addWidget(self.dataset_info_label)

        browser_group = QGroupBox("Query Browser")
        browser_layout = QVBoxLayout(browser_group)
        browser_layout.setSpacing(10)
        left_layout.addWidget(browser_group, stretch=1)

        self.sample_filter = QLineEdit()
        self.sample_filter.setPlaceholderText("Filter by class, stem, or description")
        self.sample_filter.textChanged.connect(self._refresh_sample_list)
        browser_layout.addWidget(self.sample_filter)

        filter_actions = QHBoxLayout()
        filter_actions.setSpacing(8)
        browser_layout.addLayout(filter_actions)

        self.match_count_label = QLabel("0 matches")
        self.match_count_label.setObjectName("mutedText")
        filter_actions.addWidget(self.match_count_label)

        filter_actions.addStretch(1)

        self.clear_filter_button = QPushButton("Clear")
        self.clear_filter_button.clicked.connect(self.sample_filter.clear)
        filter_actions.addWidget(self.clear_filter_button)

        self.random_button = QPushButton("Random")
        self.random_button.clicked.connect(self._pick_random_sample)
        filter_actions.addWidget(self.random_button)

        self.sample_list = QListWidget()
        self.sample_list.setSelectionMode(QAbstractItemView.SingleSelection)
        self.sample_list.setTextElideMode(Qt.ElideRight)
        self.sample_list.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.sample_list.itemSelectionChanged.connect(self._on_sample_changed)
        browser_layout.addWidget(self.sample_list, stretch=1)

        self.sample_info_label = QLabel("Select a query sample.")
        self.sample_info_label.setWordWrap(True)
        self.sample_info_label.setObjectName("mutedText")
        browser_layout.addWidget(self.sample_info_label)

        query_group = QGroupBox("Query Setup")
        query_layout = QGridLayout(query_group)
        query_layout.setHorizontalSpacing(12)
        query_layout.setVerticalSpacing(12)
        query_layout.setColumnStretch(0, 1)
        query_layout.setColumnStretch(1, 1)
        query_layout.setColumnStretch(2, 1)
        right_layout.addWidget(query_group, 3)

        self.selected_sketch_card = ImageCard("Selected Sketch", QSize(200, 180))
        query_layout.addWidget(self.selected_sketch_card, 0, 0)

        self.reference_card = ImageCard("Reference Image", QSize(200, 180))
        query_layout.addWidget(self.reference_card, 0, 1)

        prompt_panel = QWidget()
        prompt_layout = QVBoxLayout(prompt_panel)
        prompt_layout.setContentsMargins(0, 0, 0, 0)
        prompt_layout.setSpacing(10)
        query_layout.addWidget(prompt_panel, 0, 2)

        self.query_title_label = QLabel("No sample selected")
        self.query_title_label.setObjectName("sectionTitle")
        self.query_title_label.setWordWrap(True)
        prompt_layout.addWidget(self.query_title_label)

        self.query_meta_label = QLabel("")
        self.query_meta_label.setObjectName("mutedText")
        self.query_meta_label.setWordWrap(True)
        prompt_layout.addWidget(self.query_meta_label)

        self.text_edit = QPlainTextEdit()
        self.text_edit.setPlaceholderText("Describe the target image in natural language...")
        self.text_edit.setMinimumHeight(170)
        prompt_layout.addWidget(self.text_edit, stretch=1)

        prompt_actions = QHBoxLayout()
        prompt_actions.setSpacing(10)
        prompt_layout.addLayout(prompt_actions)

        self.restore_prompt_button = QPushButton("Use Default Prompt")
        self.restore_prompt_button.clicked.connect(self._restore_original_prompt)
        self.restore_prompt_button.setMinimumWidth(170)
        prompt_actions.addWidget(self.restore_prompt_button)

        prompt_actions.addStretch(1)

        self.retrieve_button = QPushButton("Retrieve Top-5")
        self.retrieve_button.clicked.connect(self._start_retrieval)
        self.retrieve_button.setMinimumWidth(170)
        prompt_actions.addWidget(self.retrieve_button)

        self.result_info_label = QLabel("Run a retrieval to see the ranked results.")
        self.result_info_label.setObjectName("mutedText")
        self.result_info_label.setWordWrap(True)
        prompt_layout.addWidget(self.result_info_label)

        sketch_group = QGroupBox("Sketch Candidates")
        sketch_layout = QVBoxLayout(sketch_group)
        sketch_layout.setSpacing(8)
        right_layout.addWidget(sketch_group, 2)

        self.sketch_list = QListWidget()
        self.sketch_list.setViewMode(QListWidget.IconMode)
        self.sketch_list.setFlow(QListWidget.LeftToRight)
        self.sketch_list.setWrapping(False)
        self.sketch_list.setResizeMode(QListWidget.Adjust)
        self.sketch_list.setMovement(QListWidget.Static)
        self.sketch_list.setSelectionMode(QAbstractItemView.SingleSelection)
        self.sketch_list.setIconSize(QSize(118, 118))
        self.sketch_list.setGridSize(QSize(132, 144))
        self.sketch_list.setSpacing(10)
        self.sketch_list.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.sketch_list.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.sketch_list.setMinimumHeight(176)
        self.sketch_list.setMaximumHeight(176)
        self.sketch_list.itemSelectionChanged.connect(self._on_sketch_selection_changed)
        sketch_layout.addWidget(self.sketch_list)

        self.sketch_hint = QLabel("Choose one sketch thumbnail.")
        self.sketch_hint.setObjectName("mutedText")
        sketch_layout.addWidget(self.sketch_hint)

        results_group = QGroupBox("Retrieval Results")
        results_layout = QGridLayout(results_group)
        results_layout.setHorizontalSpacing(12)
        results_layout.setVerticalSpacing(12)
        results_layout.setColumnStretch(0, 1)
        results_layout.setColumnStretch(1, 1)
        results_layout.setColumnStretch(2, 1)
        results_layout.setColumnStretch(3, 1)
        results_layout.setColumnStretch(4, 1)
        right_layout.addWidget(results_group, 3)

        self.result_cards = []
        positions = [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)]
        for index, (row, column) in enumerate(positions, start=1):
            card = ImageCard("Top-{}".format(index), QSize(168, 188), accent=index == 1)
            self.result_cards.append(card)
            results_layout.addWidget(card, row, column)

    def _apply_styles(self) -> None:
        self.setStyleSheet(
            """
            QMainWindow, QWidget {
                background: #f2eee7;
                color: #23313b;
                font-size: 13px;
            }
            QGroupBox {
                background: #fffdf9;
                border: 1px solid #d7d0c3;
                border-radius: 14px;
                margin-top: 12px;
                font-weight: 700;
                padding-top: 14px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 14px;
                padding: 0 4px;
            }
            QLabel#subtitle {
                color: #607077;
                font-size: 14px;
            }
            QLabel#headerPill {
                background: #dcece7;
                color: #24584f;
                border: 1px solid #b7d4cb;
                border-radius: 999px;
                padding: 8px 12px;
                font-weight: 700;
            }
            QLabel#sectionTitle {
                font-size: 16px;
                font-weight: 700;
            }
            QLabel#mutedText {
                color: #637378;
            }
            QLabel#statusText {
                background: #f8f5ef;
                border: 1px solid #ddd6c8;
                border-radius: 12px;
                color: #345159;
                padding: 12px;
            }
            QLabel#cardTitle {
                font-weight: 700;
                font-size: 14px;
            }
            QLabel#imageFrame, QLabel#imageFrameAccent {
                background: #f7f3ec;
                border-radius: 12px;
                color: #6a797f;
            }
            QLabel#imageFrame {
                border: 1px solid #d8d1c6;
            }
            QLabel#imageFrameAccent {
                border: 2px solid #2e6f68;
            }
            QLabel#captionText {
                color: #5d6c72;
            }
            QComboBox, QLineEdit, QPlainTextEdit, QListWidget {
                background: #ffffff;
                border: 1px solid #cfc7b8;
                border-radius: 10px;
                padding: 8px 10px;
            }
            QListWidget::item {
                padding: 6px;
                border-radius: 8px;
            }
            QListWidget::item:selected {
                background: #dcece7;
                color: #1f4944;
                border: 1px solid #2e6f68;
            }
            QPushButton {
                background: #2e6f68;
                color: white;
                border: none;
                border-radius: 10px;
                padding: 9px 14px;
                font-weight: 700;
            }
            QPushButton:hover {
                background: #285f59;
            }
            QPushButton:disabled {
                background: #9bb1ac;
            }
            """
        )

    def closeEvent(self, event) -> None:
        for widget in (self.text_edit, self.sample_filter, self.sample_list, self.sketch_list):
            widget.clearFocus()
        focused = QApplication.focusWidget()
        if focused is not None:
            focused.clearFocus()
        super().closeEvent(event)

    def _configure_initial_geometry(self) -> None:
        default_width = 1560
        default_height = 960
        screen = QApplication.primaryScreen()
        available = screen.availableGeometry() if screen is not None else None

        if available is None:
            target_width = default_width
            target_height = default_height
        else:
            target_width = min(max(self.minimumWidth(), int(available.width() * 0.9)), 1760)
            target_height = min(max(self.minimumHeight(), int(available.height() * 0.9)), 1080)

        self.resize(target_width, target_height)

        left_width = max(360, int(target_width * 0.24))
        right_width = max(920, target_width - left_width)
        self.main_splitter.setSizes([left_width, right_width])

        if available is None:
            return

        self.move(
            available.x() + max(0, (available.width() - target_width) // 2),
            available.y() + max(0, (available.height() - target_height) // 2),
        )

    def _set_status(self, message: str) -> None:
        self.header_status.setText(message)

    def _set_controls_enabled(self, enabled: bool) -> None:
        has_session = self.current_session is not None
        self.dataset_combo.setEnabled(enabled)
        self.sample_filter.setEnabled(enabled)
        self.clear_filter_button.setEnabled(enabled)
        self.random_button.setEnabled(enabled and self.sample_list.count() > 0)
        self.sample_list.setEnabled(enabled)
        self.sketch_list.setEnabled(enabled and has_session)
        self.text_edit.setEnabled(enabled and has_session)
        self.restore_prompt_button.setEnabled(enabled and has_session)
        self.retrieve_button.setEnabled(enabled and has_session)

    def _populate_datasets(self) -> None:
        self.dataset_combo.blockSignals(True)
        self.dataset_combo.clear()
        for definition in self.workspace.list_datasets():
            self.dataset_combo.addItem(definition.label, definition.key)
        self.dataset_combo.blockSignals(False)

    def _clear_results(self) -> None:
        for index, card in enumerate(self.result_cards, start=1):
            card.clear("Top-{}".format(index))

    def _clear_query_view(self) -> None:
        self.sample_list.clear()
        self.sample_info_label.setText("Select a query sample.")
        self.match_count_label.setText("0 matches")
        self.query_title_label.setText("No sample selected")
        self.query_meta_label.setText("")
        self.text_edit.clear()
        self.sketch_list.clear()
        self.sketch_hint.setText("Choose one sketch thumbnail.")
        self.selected_sketch_card.clear("Selected Sketch")
        self.reference_card.clear("Reference Image")
        self.result_info_label.setText("Run a retrieval to see the ranked results.")
        self._clear_results()

    def _start_loading_current_dataset(self) -> None:
        dataset_key = self.dataset_combo.currentData()
        if not dataset_key:
            return

        self.current_session = None
        self.dataset_info_label.setText("Loading dataset: {}".format(dataset_key))
        self._clear_query_view()
        self._set_controls_enabled(False)
        self._set_status("Loading {} resources...".format(dataset_key))

        self._load_thread = DatasetLoaderThread(self.workspace, dataset_key)
        self._load_thread.status.connect(self._set_status)
        self._load_thread.loaded.connect(self._on_dataset_loaded)
        self._load_thread.failed.connect(self._on_worker_failed)
        self._load_thread.start()

    def _on_dataset_loaded(self, session: FusionDatasetSession) -> None:
        expected_key = self.dataset_combo.currentData()
        if session.definition.key != expected_key:
            return

        self.current_session = session
        self.dataset_info_label.setText(
            "{} | {} samples | gamma {:.2f} | device {}".format(
                session.definition.label,
                len(session.samples),
                session.gamma,
                session.device_name,
            )
        )
        self._refresh_sample_list()
        self._set_controls_enabled(True)
        self._set_status("Dataset ready: {}".format(session.definition.label))

    def _refresh_sample_list(self) -> None:
        if self.current_session is None:
            return

        query = self.sample_filter.text().strip().lower()
        matches: list[QuerySample] = []
        for sample in self.current_session.samples:
            haystack = " ".join(
                [
                    sample.label,
                    sample.virtual_class,
                    sample.category,
                    sample.id,
                    sample.text,
                    sample.image_path.name,
                ]
            ).lower()
            if query and query not in haystack:
                continue
            matches.append(sample)

        self.sample_list.blockSignals(True)
        self.sample_list.clear()
        for sample in matches:
            item = QListWidgetItem(sample.label)
            item.setData(Qt.UserRole, sample)
            item.setToolTip(sample.text)
            self.sample_list.addItem(item)
        self.sample_list.blockSignals(False)

        self.match_count_label.setText("{} / {} matches".format(len(matches), len(self.current_session.samples)))

        if matches:
            self.sample_list.setCurrentRow(0)
            self._on_sample_changed()
        else:
            self.sample_info_label.setText("No samples match the current filter.")
            self.query_title_label.setText("No sample selected")
            self.query_meta_label.setText("")
            self.text_edit.clear()
            self.sketch_list.clear()
            self.selected_sketch_card.clear("Selected Sketch")
            self.reference_card.clear("Reference Image")
            self.result_info_label.setText("Run a retrieval to see the ranked results.")
            self._clear_results()

    def _pick_random_sample(self) -> None:
        if self.sample_list.count() == 0:
            return
        self.sample_list.setCurrentRow(random.randrange(self.sample_list.count()))

    def _current_sample(self) -> Optional[QuerySample]:
        selected = self.sample_list.selectedItems()
        if not selected:
            return None
        data = selected[0].data(Qt.UserRole)
        if isinstance(data, QuerySample):
            return data
        return None

    def _on_sample_changed(self) -> None:
        sample = self._current_sample()
        if sample is None:
            return

        self.sample_info_label.setText(
            "ID: {} | {} sketch candidates | reference {}".format(
                sample.id,
                len(sample.sketch_paths),
                sample.image_path.name,
            )
        )
        self.query_title_label.setText(sample.virtual_class)
        self.query_meta_label.setText("Reference image: {}".format(sample.image_path.name))
        self.text_edit.setPlainText(sample.text)
        self.reference_card.set_content("Reference Image", sample.image_path, caption=sample.image_path.name)
        self._populate_sketches(sample)
        self.result_info_label.setText("Edit the prompt if needed, then retrieve the top-5 images.")
        self._clear_results()

    def _populate_sketches(self, sample: QuerySample) -> None:
        self.sketch_list.clear()
        for index, sketch_path in enumerate(sample.sketch_paths):
            item = QListWidgetItem("Sketch {}".format(index + 1))
            item.setData(Qt.UserRole, sketch_path)
            item.setToolTip(sketch_path.name)
            icon = QIcon(str(sketch_path))
            if not icon.isNull():
                item.setIcon(icon)
            self.sketch_list.addItem(item)

        if self.sketch_list.count() > 0:
            self.sketch_list.setCurrentRow(0)
            self._on_sketch_selection_changed()
        else:
            self.selected_sketch_card.clear("Selected Sketch")
            self.sketch_hint.setText("No sketch candidates found for this sample.")

    def _selected_sketch_path(self) -> Optional[Path]:
        selected_items = self.sketch_list.selectedItems()
        if not selected_items:
            return None
        value = selected_items[0].data(Qt.UserRole)
        if isinstance(value, Path):
            return value
        if value:
            return Path(value)
        return None

    def _on_sketch_selection_changed(self) -> None:
        sketch_path = self._selected_sketch_path()
        if sketch_path is None:
            self.selected_sketch_card.clear("Selected Sketch")
            self.sketch_hint.setText("Choose one sketch thumbnail.")
            return

        self.selected_sketch_card.set_content("Selected Sketch", sketch_path, caption=sketch_path.name)
        self.sketch_hint.setText("Using sketch: {}".format(sketch_path.name))

    def _restore_original_prompt(self) -> None:
        sample = self._current_sample()
        if sample is not None:
            self.text_edit.setPlainText(sample.text)

    def _start_retrieval(self) -> None:
        if self.current_session is None:
            QMessageBox.warning(self, "Dataset Not Ready", "Please wait until the dataset has finished loading.")
            return

        sample = self._current_sample()
        sketch_path = self._selected_sketch_path()
        text = self.text_edit.toPlainText().strip()

        if sample is None:
            QMessageBox.warning(self, "No Sample", "Please select a query sample first.")
            return
        if sketch_path is None:
            QMessageBox.warning(self, "No Sketch", "Please choose one sketch candidate.")
            return
        if not text:
            QMessageBox.warning(self, "Empty Text", "Please enter a text prompt.")
            return

        self._set_controls_enabled(False)
        self.result_info_label.setText("Retrieval in progress...")
        self._retrieval_thread = RetrievalThread(
            session=self.current_session,
            sample=sample,
            sketch_path=sketch_path,
            text=text,
            top_k=5,
        )
        self._retrieval_thread.status.connect(self._set_status)
        self._retrieval_thread.completed.connect(self._on_retrieval_completed)
        self._retrieval_thread.failed.connect(self._on_worker_failed)
        self._retrieval_thread.start()

    def _on_retrieval_completed(self, result: RetrievalResult) -> None:
        for index, card in enumerate(self.result_cards):
            if index < len(result.hits):
                hit = result.hits[index]
                card.set_content(
                    "Top-{} | {:.4f}".format(hit.rank, hit.score),
                    hit.image_path,
                    caption=hit.image_path.name,
                )
            else:
                card.clear("Top-{}".format(index + 1))

        gt_text = "unknown" if result.ground_truth_rank is None else str(result.ground_truth_rank)
        self.result_info_label.setText(
            "Ground-truth rank: {} | Query sample: {} | Prompt length: {} characters".format(
                gt_text,
                result.sample.id,
                len(result.query_text),
            )
        )
        self._set_status("Retrieved top-5 results for {}".format(result.sample.id))
        self._set_controls_enabled(True)

    def _on_worker_failed(self, traceback_text: str) -> None:
        self._set_controls_enabled(True)
        self._set_status("Operation failed")
        QMessageBox.critical(self, "Error", traceback_text)


def launch_app() -> int:
    app = QApplication.instance()
    owns_app = app is None
    if app is None:
        app = QApplication([])
    font_db = QFontDatabase()
    installed = set(font_db.families())
    for family in (
        "PingFang SC",
        "Helvetica Neue",
        "Arial Unicode MS",
        "Segoe UI",
        "Microsoft YaHei UI",
        "Noto Sans CJK SC",
    ):
        if family in installed:
            app.setFont(QFont(family, 13))
            break
    window = MainWindow()
    window.show()
    if owns_app:
        return app.exec_()
    return 0
