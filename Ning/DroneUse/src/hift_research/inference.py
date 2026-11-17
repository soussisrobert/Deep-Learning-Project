"""Original GPU model and opt-in score-only adapter, without upstream file changes."""
from pathlib import Path
import sys


def configure_upstream(root, config=None):
    from hift_research.provenance import source_identity
    source_identity(root)  # fail closed on upstream drift
    upstream = Path(root).resolve() / 'third_party/HiFT'
    if str(upstream) not in sys.path:
        sys.path.insert(0, str(upstream))
    from pysot.core.config import cfg
    cfg.defrost()
    cfg.merge_from_file(str(config or upstream / 'experiments/config.yaml'))
    cfg.CUDA = True
    return cfg


def read_checkpoint_state(checkpoint):
    """Restricted tensor load with only the released file's NumPy scalar metadata types.

    Official file also contains optimizer state; never execute an unrestricted pickle.
    Old NumPy import path is explicitly mapped to the current scalar constructor.
    """
    import numpy as np
    import torch
    safe = [np._core.multiarray.scalar, (np._core.multiarray.scalar, 'numpy.core.multiarray.scalar'),
            np.dtype, np.dtypes.Float64DType]
    with torch.serialization.safe_globals(safe):
        state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    if not isinstance(state, dict):
        raise ValueError('Expected a state_dict mapping')
    state = state.get('state_dict', state)
    if not isinstance(state, dict) or not all(isinstance(k, str) and isinstance(v, torch.Tensor) for k, v in state.items()):
        raise ValueError('Checkpoint must contain tensor weights only')
    keys = [k.removeprefix('module.') for k in state]
    if len(set(keys)) != len(keys):
        raise ValueError('Duplicate checkpoint keys after prefix normalization')
    return dict(zip(keys, state.values()))


def load_original(root, checkpoint=None, config=None):
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('Original HiFT hardcodes CUDA; CPU mathematical tests remain available. No silent CPU port.')
    configure_upstream(root, config)
    from pysot.models.model_builder import ModelBuilder
    model = ModelBuilder().cuda().eval()
    if checkpoint is not None:
        model.load_state_dict(read_checkpoint_state(checkpoint), strict=True)
    return model


class ScoreProbabilityAdapter:
    """B1-score only: sigmoid BCE logits before original tracker consumes cls2.

    This corrects the output interpretation, not statistical probability calibration.
    Layout, model weights, penalties, crop geometry, clipping remain unchanged.
    """
    def __init__(self, model):
        self.model = model

    def eval(self):
        self.model.eval()
        return self

    def template(self, image):
        return self.model.template(image)

    def track(self, image):
        result = dict(self.model.track(image))
        result['cls2'] = result['cls2'].sigmoid()
        return result


def make_tracker(model, baseline='B0'):
    from pysot.tracker.hift_tracker import HiFTTracker
    if baseline == 'B1-score':
        model = ScoreProbabilityAdapter(model)
    elif baseline != 'B0':
        raise ValueError('Only B0 and B1-score inference is enabled; B2/R1 require validated training')
    return HiFTTracker(model)
