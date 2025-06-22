import threading
import time
import numpy as np

# Attempt to import BrainFlow; fallback to DummyBoard if missing
try:
    from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
except (ImportError, FileNotFoundError):
    BoardShim = BrainFlowInputParams = BoardIds = None


class DummyBoard:
    """Minimal dummy board streaming zeros for 4 channels."""

    def __init__(self, *args, **kwargs):
        pass

    def prepare_session(self):
        pass

    def start_stream(self, packet_size):
        pass

    def get_current_board_data(self, n_samples):
        return np.zeros((4, n_samples))

    def stop_stream(self):
        pass

    def release_session(self):
        pass


class BCIClient:
    def __init__(self, board_id=None, serial_port=None, use_dummy: bool = True):
        """Initialize BCIClient; default to dummy mode."""
        self.use_dummy = use_dummy
        self._buffer = []
        self._stream_thread = None
        self._keep_streaming = False
        self._processing = False

        # Configure board
        params = (
            None
            if use_dummy
            else (BrainFlowInputParams() if BrainFlowInputParams else None)
        )
        if params and serial_port:
            params.serial_port = serial_port
        if use_dummy or not BoardShim or not params:
            self.board = DummyBoard()
        else:
            try:
                bid = board_id or BoardIds.CYTON_BOARD.value
                self.board = BoardShim(bid, params)
            except Exception:
                self.board = DummyBoard()

    def connect(self):
        """Prepare session (no-op in dummy)."""
        self.board.prepare_session()

    def disconnect(self):
        """Release session (no-op in dummy)."""
        self.board.release_session()

    def start_stream(self, sampling_rate=250, packet_size=4500):
        """Begin streaming raw data into internal buffer."""
        self.board.start_stream(packet_size)
        self._keep_streaming = True

        def _read_loop():
            while self._keep_streaming:
                data = self.board.get_current_board_data(sampling_rate)
                self._buffer.append(data)
                time.sleep(1.0 / sampling_rate)

        self._stream_thread = threading.Thread(target=_read_loop, daemon=True)
        self._stream_thread.start()

    def stop_stream(self):
        """Stop streaming and release session."""
        self._keep_streaming = False
        if self._stream_thread is not None:
            self._stream_thread.join()
        self.board.stop_stream()
        self.board.release_session()

    def get_buffer(self):
        """Retrieve and clear the accumulated data buffer."""
        if self._buffer:
            buf = np.hstack(self._buffer)
        else:
            buf = np.empty((0, 0))
        self._buffer.clear()
        return buf

    def get_features(self, fs, preprocess_fn, feature_fn):
        """Convenience: buffer→preprocess→features."""
        raw = self.get_buffer()
        if raw.size == 0:
            return {}
        clean = preprocess_fn(raw, fs)
        return feature_fn(clean, fs)

    def predict(self, fs, preprocess_fn, feature_fn, decode_fn, decoding_spec):
        """Convenience: features→decode."""
        feats = self.get_features(fs, preprocess_fn, feature_fn)
        return decode_fn(feats, decoding_spec)

    def start_processing_pipeline(
        self,
        fs: int,
        preprocess_fn,
        feature_fn,
        decode_fn,
        decoding_spec: dict,
        window_size_s: float = 1.0,
        step_size_s: float = 0.5,
        callback=None,
    ):
        """Stream data and process in sliding windows."""
        self._processing = True
        window_samples = int(window_size_s * fs)
        step_samples = int(step_size_s * fs)

        def _proc_loop():
            while self._processing:
                all_data = np.hstack(self._buffer) if self._buffer else np.empty((0, 0))
                if all_data.size == 0:
                    time.sleep(step_size_s)
                    continue
                ch, tot = all_data.shape
                idx = 0
                while idx + window_samples <= tot:
                    win = all_data[:, idx : idx + window_samples]
                    ts = idx / fs
                    clean = preprocess_fn(win, fs)
                    feats = feature_fn(clean, fs)
                    preds = decode_fn(feats, decoding_spec)
                    if callback:
                        callback(preds, ts)
                    idx += step_samples
                if idx > 0:
                    tail = all_data[:, idx:]
                    self._buffer = [tail]
                time.sleep(step_size_s)

        # Start thread without launching hardware stream in dummy mode
        if not self.use_dummy:
            self.start_stream(sampling_rate=fs, packet_size=fs * 2)
        self._processing_thread = threading.Thread(target=_proc_loop, daemon=True)
        self._processing_thread.start()

    def stop_processing_pipeline(self):
        """Stop processing loop and hardware stream."""
        self._processing = False
        if self._processing_thread is not None:
            self._processing_thread.join()
        if not self.use_dummy:
            self.stop_stream()
