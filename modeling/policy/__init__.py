from .denoise_actor_3d import DenoiseActor as DenoiseActor3D
# Note: denoise_actor_2d removed - not used in this project (3D only)
from .bimanual_composer import BimanualComposer

# Import EBM components (with fallback if not available)
try:
    from ..ebm_compositionality import EBMBimanualComposer
    EBM_AVAILABLE = True
except ImportError:
    EBM_AVAILABLE = False
    EBMBimanualComposer = None
    print("Warning: EBM components not available. Use original BimanualComposer instead.")


def fetch_model_class(model_type):
    if model_type == 'denoise3d':  # standard 3DFA
        return DenoiseActor3D
    if model_type == 'denoise2d':  # 2D model not available in this version
        raise NotImplementedError("2D model has been removed. Use 'denoise3d' instead.")
    if model_type == 'bimanual_composer':
        return BimanualComposer
    if model_type == 'ebm_bimanual_composer':
        if EBM_AVAILABLE:
            return EBMBimanualComposer
        else:
            print("Warning: EBM not available, falling back to BimanualComposer")
            return BimanualComposer
    return None
