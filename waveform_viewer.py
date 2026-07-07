import sys
import numpy as np
import soundfile as sf
from PyQt6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QLabel, QPushButton, QHBoxLayout, QSlider
from PyQt6.QtCore import Qt, QPointF, QUrl, QTimer, QElapsedTimer
from PyQt6.QtMultimedia import QMediaPlayer, QAudioOutput
from PyQt6.QtGui import QColor, QPalette
import pyqtgraph as pg

# Configure pyqtgraph for dark mode
pg.setConfigOption('background', '#121212')
pg.setConfigOption('foreground', '#E0E0E0')
pg.setConfigOptions(antialias=False) # Antialiasing causes heavy lag on large waveform datasets

class BreathRegionItem(pg.LinearRegionItem):
    def __init__(self, viewer, *args, **kwargs):
        # Default pens, otherwise pyqtgraph uses yellow
        if 'pen' not in kwargs:
            kwargs['pen'] = pg.mkPen('#03DAC6', width=1)
        if 'hoverPen' not in kwargs:
            # Thick hover pen widens the grab area
            kwargs['hoverPen'] = pg.mkPen(color='#00B3A6', width=12)
            
        super().__init__(*args, **kwargs)
        self.viewer = viewer
        
        # Apply the hover pen to both edges
        for line in self.lines:
            line.setHoverPen(kwargs['hoverPen'])
        
        self.default_brush = pg.mkBrush(3, 218, 198, 40)
        self.hover_brush = pg.mkBrush(3, 218, 198, 80)
        self.selected_brush = pg.mkBrush(3, 218, 198, 120) # Brighter for selected
        
        self.setBrush(self.default_brush)
        self.setHoverBrush(self.hover_brush)
        self.is_selected = False

    def set_selected(self, selected):
        self.is_selected = selected
        if selected:
            self.setBrush(self.selected_brush)
            self.setHoverBrush(self.selected_brush)
            
            start, end = self.getRegion()
            center = (start + end) / 2
            view_box = self.viewer.plot_widget.getViewBox()
            x_min, x_max = view_box.viewRange()[0]
            range_width = x_max - x_min
            self.viewer.plot_widget.setXRange(center - range_width/2, center + range_width/2, padding=0)
        else:
            self.setBrush(self.default_brush)
            self.setHoverBrush(self.hover_brush)

    def mouseClickEvent(self, ev):
        if ev.button() == Qt.MouseButton.LeftButton:
            ev.accept()
            self.viewer.select_region(self)
        else:
            super().mouseClickEvent(ev)

class TimeAxisItem(pg.AxisItem):
    def tickStrings(self, values, scale, spacing):
        strings = []
        for v in values:
            if v < 0:
                strings.append("")
                continue
            
            # m:ss over a minute, plain seconds otherwise
            minutes = int(v) // 60
            seconds = int(v) % 60
            
            if minutes > 0:
                strings.append(f"{minutes}:{seconds:02d}")
            else:
                strings.append(f"{seconds}")
        return strings

class CustomPlotWidget(pg.PlotWidget):
    def __init__(self, viewer=None, **kwargs):
        # Use the custom time axis
        kwargs['axisItems'] = {'bottom': TimeAxisItem(orientation='bottom')}
        super().__init__(**kwargs)
        self.viewer = viewer

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.MiddleButton:
            self.viewer.is_scrubbing = True
            self.viewer.last_mouse_x = event.position().x()
        elif event.button() == Qt.MouseButton.LeftButton:
            view_point = self.getViewBox().mapSceneToView(event.position())
            self.viewer.set_playhead(view_point.x())
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.MiddleButton:
            self.viewer.is_scrubbing = False
            self.viewer.last_mouse_x = None
        super().mouseReleaseEvent(event)

    def mouseMoveEvent(self, event):
        if self.viewer.is_scrubbing and self.viewer.last_mouse_x is not None:
            current_x = event.position().x()
            dx = current_x - self.viewer.last_mouse_x
            self.viewer.last_mouse_x = current_x
            
            view_box = self.getViewBox()
            view_range = view_box.viewRange()
            x_min, x_max = view_range[0]
            
            range_width = x_max - x_min
            
            # Drag right shifts the view left
            shift = -dx * (range_width / self.width())
            
            new_min = x_min + shift
            new_max = x_max + shift
            
            # Don't scrub past the edges
            if new_min < 0:
                new_max -= new_min
                new_min = 0
            if new_max > self.viewer.total_time:
                new_min -= (new_max - self.viewer.total_time)
                new_max = self.viewer.total_time
                if new_min < 0:
                    new_min = 0
                
            self.setXRange(new_min, new_max, padding=0)
            
        super().mouseMoveEvent(event)

class WaveformViewer(QMainWindow):
    def __init__(self, audio_path=None):
        super().__init__()
        self.setWindowTitle("Waveform Viewer")
        self.resize(1000, 300)
        self.setStyleSheet("QMainWindow { background-color: #121212; }")
        
        self.central_widget = QWidget()
        self.central_widget.setStyleSheet("background-color: #121212;")
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)
        self.layout.setContentsMargins(20, 20, 20, 20)
        self.layout.setSpacing(15)
        
        # Audio setup must be done before connecting the slider
        self.audio_output = QAudioOutput()
        self.player = QMediaPlayer()
        self.player.setAudioOutput(self.audio_output)

        # Volume slider
        self.top_layout = QHBoxLayout()
        
        self.volume_label = QLabel("Volume (100%)")
        self.volume_label.setStyleSheet("color: #E0E0E0; font-weight: bold; font-family: 'Segoe UI', 'Inter', sans-serif;")
        self.top_layout.addWidget(self.volume_label)
        
        self.volume_slider = QSlider(Qt.Orientation.Horizontal)
        self.volume_slider.setRange(0, 500) # 0 to 500%
        self.volume_slider.setValue(100) # Default 100%
        self.volume_slider.setFixedWidth(200)
        self.volume_slider.setStyleSheet("""
            QSlider::groove:horizontal {
                border: 1px solid #333333;
                height: 8px;
                background: #1E1E1E;
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background: #03DAC6;
                width: 16px;
                height: 16px;
                margin: -4px 0;
                border-radius: 8px;
            }
        """)
        self.volume_slider.valueChanged.connect(self.on_volume_changed)
        self.volume_slider.sliderReleased.connect(self.apply_volume_boost)
        self.top_layout.addWidget(self.volume_slider)
        
        self.top_layout.addStretch() # Push to the left
        self.layout.addLayout(self.top_layout)
        
        self.plot_widget = CustomPlotWidget(viewer=self)
        self.plot_widget.setStyleSheet("border: 1px solid #333333; border-radius: 8px;")
        self.layout.addWidget(self.plot_widget)
        
        self.button_layout = QHBoxLayout()
        self.button_layout.addStretch()
        
        self.check_button = QPushButton("✔ Accept")
        self.check_button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.check_button.setStyleSheet("""
            QPushButton {
                color: #03DAC6;
                background-color: #1E1E1E;
                font-size: 16px; 
                font-weight: bold; 
                border: 2px solid #03DAC6; 
                border-radius: 8px;
                padding: 10px 30px;
            }
            QPushButton:hover {
                background-color: rgba(3, 218, 198, 0.15);
            }
            QPushButton:pressed {
                background-color: rgba(3, 218, 198, 0.3);
            }
        """)
        self.check_button.setCursor(Qt.CursorShape.PointingHandCursor)
        
        self.cross_button = QPushButton("✖ Reject")
        self.cross_button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.cross_button.setStyleSheet("""
            QPushButton {
                color: #CF6679;
                background-color: #1E1E1E;
                font-size: 16px; 
                font-weight: bold; 
                border: 2px solid #CF6679; 
                border-radius: 8px;
                padding: 10px 30px;
            }
            QPushButton:hover {
                background-color: rgba(207, 102, 121, 0.15);
            }
            QPushButton:pressed {
                background-color: rgba(207, 102, 121, 0.3);
            }
        """)
        self.cross_button.setCursor(Qt.CursorShape.PointingHandCursor)
        
        self.check_button.clicked.connect(self.approve_label)
        self.cross_button.clicked.connect(self.reject_label)
        
        self.button_layout.addWidget(self.check_button)
        self.button_layout.addSpacing(20)
        self.button_layout.addWidget(self.cross_button)
        
        self.button_layout.addStretch()
        self.layout.addLayout(self.button_layout)
        
        # Disable default mouse handling; middle mouse is handled manually
        self.plot_widget.setMouseEnabled(x=False, y=False)
        self.plot_widget.hideAxis('left')
        
        self.setAcceptDrops(True)
        
        self.total_time = 0
        self.is_scrubbing = False
        self.last_mouse_x = None
        self.regions = []
        self.selected_region = None

        
        # Playhead line
        self.playhead_line = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen(color='#FF0266', width=2))
        self.playhead_line.setZValue(1000)  # draw above the waveform
        self.plot_widget.addItem(self.playhead_line)
        self.playhead_pos = 0.0
        
        # Smooth playback state
        self.play_timer = QElapsedTimer()
        self.last_reported_pos = 0.0
        
        self.update_timer = QTimer()
        self.update_timer.setInterval(16) # ~60 fps
        self.update_timer.timeout.connect(self.update_playhead)

        if audio_path:
            self.load_audio(audio_path)
            
    def on_volume_changed(self, value):
        self.volume_label.setText(f"Volume ({value}%)")

    def apply_volume_boost(self):
        if not hasattr(self, 'current_audio_path') or not self.current_audio_path:
            return
            
        value = self.volume_slider.value()
        multiplier = value / 100.0
        
        if multiplier == 1.0:
            # 100%: use the original file
            is_playing = self.player.playbackState() == QMediaPlayer.PlaybackState.PlayingState
            pos = self.player.position()
            self.player.setSource(QUrl.fromLocalFile(self.current_audio_path))
            self.player.setPosition(pos)
            if is_playing:
                self.player.play()
            return
            
        try:
            import soundfile as sf
            import numpy as np
            import tempfile
            import os
            
            data, samplerate = sf.read(self.current_audio_path)
            # Clip to prevent wrap-around distortion
            data = np.clip(data * multiplier, -1.0, 1.0)
            
            temp_dir = tempfile.gettempdir()
            current_basename = os.path.basename(self.current_audio_path)
            self.boosted_audio_path = os.path.join(temp_dir, f"boosted_{current_basename}")
            
            sf.write(self.boosted_audio_path, data, samplerate)
            
            # Switch source smoothly
            is_playing = self.player.playbackState() == QMediaPlayer.PlaybackState.PlayingState
            pos = self.player.position()
            self.player.setSource(QUrl.fromLocalFile(self.boosted_audio_path))
            self.player.setPosition(pos)
            if is_playing:
                self.player.play()
                
        except Exception as e:
            print(f"Error boosting volume: {e}")

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Space:
            self.toggle_playback()
            return
        super().keyPressEvent(event)
        
    def toggle_playback(self):
        if self.player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            self.player.pause()
            self.update_timer.stop()
        else:
            if self.player.mediaStatus() == QMediaPlayer.MediaStatus.NoMedia:
                return
            # Restart if at the end
            if self.playhead_pos >= self.total_time - 0.1:
                self.set_playhead(0.0)
            self.player.play()
            self.last_reported_pos = self.player.position() / 1000.0
            self.play_timer.restart()
            self.update_timer.start()

    def select_region(self, region):
        for r in self.regions:
            r.set_selected(False)
        self.selected_region = region
        if region:
            region.set_selected(True)
            start, _ = region.getRegion()
            self.set_playhead(start)
            
    def set_playhead(self, x_pos):
        x_pos = max(0.0, min(float(x_pos), float(self.total_time)))
        self.playhead_pos = x_pos
        self.playhead_line.setValue(x_pos)
        self.player.setPosition(int(x_pos * 1000))
        self.last_reported_pos = x_pos
        self.play_timer.restart()
        
    def update_playhead(self):
        if self.player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            pos_ms = self.player.position()
            current_reported_pos = pos_ms / 1000.0
            
            # The player reports position in chunks; interpolate between them
            pos_diff = abs(current_reported_pos - self.last_reported_pos)
            if pos_diff > 0.05: # Player moved significantly, resync
                self.last_reported_pos = current_reported_pos
                self.play_timer.restart()
            
            # Interpolate from elapsed time since last sync
            self.playhead_pos = self.last_reported_pos + (self.play_timer.elapsed() / 1000.0)
            
            # Don't run past the end before the player stops
            self.playhead_pos = min(self.playhead_pos, self.total_time)
            
            self.playhead_line.setValue(self.playhead_pos)
            
            # Auto-scroll when the playhead hits the right edge
            view_box = self.plot_widget.getViewBox()
            x_min, x_max = view_box.viewRange()[0]
            if self.playhead_pos > x_max:
                range_width = x_max - x_min
                self.plot_widget.setXRange(self.playhead_pos, self.playhead_pos + range_width, padding=0)

    def approve_label(self):
        if self.selected_region and hasattr(self, 'current_audio_path'):
            import os
            start, end = self.selected_region.getRegion()
            
            base_name = os.path.splitext(os.path.basename(self.current_audio_path))[0]
            script_dir = os.path.dirname(os.path.abspath(__file__))
            reviewed_dir = os.path.join(script_dir, "Reviewed")
            os.makedirs(reviewed_dir, exist_ok=True)
            
            reviewed_file = os.path.join(reviewed_dir, f"{base_name}.txt")
            with open(reviewed_file, "a", encoding="utf-8") as f:
                f.write(f"{start:.3f}\t{end:.3f}\tBreath\n")
            
            self.remove_region(self.selected_region)

    def reject_label(self):
        if self.selected_region:
            self.remove_region(self.selected_region)
            
    def remove_region(self, region):
        next_region = None
        if region in self.regions:
            idx = self.regions.index(region)
            if idx + 1 < len(self.regions):
                next_region = self.regions[idx + 1]
            elif len(self.regions) > 1:
                next_region = self.regions[0]
                
        self.plot_widget.removeItem(region)
        if region in self.regions:
            self.regions.remove(region)
            
        if self.selected_region == region:
            self.selected_region = None
            if next_region:
                self.select_region(next_region)
            elif len(self.regions) == 0:
                self.show_fully_reviewed_and_close()

    def show_fully_reviewed_and_close(self):
        from PyQt6.QtWidgets import QGraphicsDropShadowEffect
        self.finished_label = QLabel("Fully Reviewed!", self.central_widget)
        self.finished_label.setStyleSheet("""
            QLabel {
                color: #03DAC6;
                font-family: 'Segoe UI', 'Inter', sans-serif;
                font-size: 56px; 
                font-weight: 800; 
                background-color: #1E1E1E; 
                border: 2px solid #BB86FC;
                border-radius: 20px; 
                padding: 40px 80px;
            }
        """)
        
        # Drop shadow
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(30)
        shadow.setColor(QColor(0, 0, 0, 180))
        shadow.setOffset(0, 10)
        self.finished_label.setGraphicsEffect(shadow)
        
        self.finished_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.finished_label.adjustSize()
        
        cx = (self.central_widget.width() - self.finished_label.width()) // 2
        cy = (self.central_widget.height() - self.finished_label.height()) // 2
        self.finished_label.move(cx, cy)
        self.finished_label.show()
        self.finished_label.raise_()
        
        self.player.stop()
        
        # Clean up Inference_Output
        import os, shutil, subprocess
        script_dir = os.path.dirname(os.path.abspath(__file__))
        inference_dir = os.path.join(script_dir, "Inference_Output")
        if os.path.exists(inference_dir):
            try:
                shutil.rmtree(inference_dir, ignore_errors=True)
            except Exception:
                pass
                
        # Run edit_audio.py once the review finishes
        edit_audio_script = os.path.join(script_dir, "edit_audio.py")
        if os.path.exists(edit_audio_script):
            print("Launching edit_audio.py...")
            subprocess.Popen([sys.executable, edit_audio_script], cwd=script_dir)
                
        QTimer.singleShot(2000, self.close)

    def load_audio(self, path):
        self.current_audio_path = path
        try:
            data, samplerate = sf.read(path)
            
            # Mix stereo to mono
            if len(data.shape) > 1:
                data = data.mean(axis=1)
                
            # Downsample; millions of points is slow to draw
            target_sr = 1000  # plenty for display
            if samplerate > target_sr:
                factor = samplerate // target_sr
                data = data[::factor]
                samplerate = target_sr
                
            time_axis = np.arange(len(data)) / samplerate
            
            self.plot_widget.clear()
            self.plot_widget.plot(time_axis, data, pen=pg.mkPen('#BB86FC'))
            
            # clear() removes the playhead, so re-add it
            self.plot_widget.addItem(self.playhead_line)
            
            self.total_time = len(data) / samplerate
            
            # Start on the first 10 seconds
            display_range = min(10.0, self.total_time)
            self.plot_widget.setXRange(0, display_range, padding=0)
            self.plot_widget.setYRange(-1, 1, padding=0.1)
            
            self.setWindowTitle(f"Waveform Viewer - {path}")
            
            import os
            base_name = os.path.splitext(os.path.basename(path))[0]
            script_dir = os.path.dirname(os.path.abspath(__file__))
            label_file = os.path.join(script_dir, "Inference_Output", base_name, f"{base_name}_inference_labels.txt")
            
            if os.path.exists(label_file):
                with open(label_file, "r", encoding="utf-8") as lf:
                    for line in lf:
                        parts = line.strip().split('\t')
                        if len(parts) >= 2:
                            start_time = float(parts[0])
                            end_time = float(parts[1])
                            # Clear old regions on reload
                            if not hasattr(self, 'regions_cleared_for_load'):
                                self.regions = []
                                self.regions_cleared_for_load = True
                                
                            region = BreathRegionItem(viewer=self, values=[start_time, end_time], movable=True)
                            self.plot_widget.addItem(region)
                            self.regions.append(region)
                            
        except Exception as e:
            if hasattr(self, 'regions_cleared_for_load'):
                delattr(self, 'regions_cleared_for_load')
            import traceback
            traceback.print_exc()
            print(f"Error loading audio: {e}")

        # apply_volume_boost sets the source based on the slider
        self.apply_volume_boost()
        
        self.set_playhead(0.0)

        if self.regions:
            # Short delay so the view range updates after drawing
            QTimer.singleShot(100, lambda: self.select_region(self.regions[0]))

    # Drag and drop audio files
    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event):
        urls = event.mimeData().urls()
        if urls:
            path = urls[0].toLocalFile()
            self.load_audio(path)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    
    # Dark palette for dialogs and tooltips
    app.setStyle("Fusion")
    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window, QColor(18, 18, 18))
    palette.setColor(QPalette.ColorRole.WindowText, QColor(224, 224, 224))
    palette.setColor(QPalette.ColorRole.Base, QColor(18, 18, 18))
    palette.setColor(QPalette.ColorRole.AlternateBase, QColor(30, 30, 30))
    palette.setColor(QPalette.ColorRole.ToolTipBase, QColor(224, 224, 224))
    palette.setColor(QPalette.ColorRole.ToolTipText, QColor(18, 18, 18))
    palette.setColor(QPalette.ColorRole.Text, QColor(224, 224, 224))
    palette.setColor(QPalette.ColorRole.Button, QColor(30, 30, 30))
    palette.setColor(QPalette.ColorRole.ButtonText, QColor(224, 224, 224))
    palette.setColor(QPalette.ColorRole.BrightText, QColor(255, 0, 0))
    palette.setColor(QPalette.ColorRole.Link, QColor(3, 218, 198))
    palette.setColor(QPalette.ColorRole.Highlight, QColor(3, 218, 198))
    palette.setColor(QPalette.ColorRole.HighlightedText, QColor(0, 0, 0))
    app.setPalette(palette)
    
    # File passed as an argument (e.g. dropped onto a desktop shortcut)
    audio_file = sys.argv[1] if len(sys.argv) > 1 else None
    
    viewer = WaveformViewer(audio_file)
    viewer.show()
    
    sys.exit(app.exec())
