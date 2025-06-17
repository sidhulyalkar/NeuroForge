
# middleware/sdk/sdk.py
import threading
import time
import numpy as np

# Try importing BrainFlow; otherwise we'll use a dummy board
try:
    from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
except (ImportError, FileNotFoundError):
    BoardShim = None
    BrainFlowInputParams = None
    BoardIds = None

class DummyBoard:
    """A minimal board that pretends to stream 4 channels of zeros."""
    def __init__(self, board_id=None, params=None):
        pass
    def prepare_session(self): pass
    def start_stream(self, packet_size): pass
    def get_current_board_data(self, n_samples):
        # 4 channels × n_samples of zeros
        return np.zeros((4, n_samples))
    def stop_stream(self): pass
    def release_session(self): pass
class BCIClient:
    def __init__(self, board_id=None, serial_port=None):
        """
        Initialize BCIClient.  Always prepare:
         - self.board (real BoardShim or DummyBoard)
         - self._buffer as list
         - self._stream_thread, self._keep_streaming flags
        """
        # Always have these attributes for streaming logic/tests:
        self._buffer = []
        self._stream_thread = None
        self._keep_streaming = False

        # BrainFlow params (if available)
        params = BrainFlowInputParams() if BrainFlowInputParams else None
        if params and serial_port:
            params.serial_port = serial_port

        # Real board or dummy
        if BoardShim and params:
            try:
                bid = board_id or BoardIds.CYTON_BOARD.value
                self.board = BoardShim(bid, params)
            except Exception:
                self.board = DummyBoard(board_id, params)
        else:
            self.board = DummyBoard(board_id, params)

    def connect(self):
        self.board.prepare_session()

    def start_stream(self, sampling_rate=250, packet_size=4500):
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
        self._keep_streaming = False
        if self._stream_thread:
            self._stream_thread.join()
        self.board.stop_stream()
        self.board.release_session()

    def get_buffer(self):
        """Return and clear the accumulated data buffer as a NumPy array."""
        buf = np.hstack(self._buffer) if self._buffer else np.empty((0,0))
        self._buffer.clear()
        return buf

    def disconnect(self):
        self.stop_stream()

    # Convenience methods for pipeline integration:
    def get_features(self, fs, preprocess_fn, feature_fn):
        raw = self.get_buffer()
        # raw shape: (channels, samples)
        cleaned = preprocess_fn(raw, fs)
        feats = feature_fn(cleaned, fs)
        return feats

    def predict(self, fs, preprocess_fn, feature_fn, decode_fn, decoding_spec):
        feats = self.get_features(fs, preprocess_fn, feature_fn)
        return decode_fn(feats, decoding_spec)
