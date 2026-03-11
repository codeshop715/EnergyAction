"""
EBM Compositionality Module

This module provides Energy-Based Model (EBM) implementations that maintain
complete interface compatibility with the original BimanualComposer while
internally using energy-based optimization for bimanual action generation.

Key Components:
- FlowToEnergyConverter: Converts Flow Matching outputs to energy functions
- EnergyComposer: Combines individual arm energies into joint energy
- EBMBimanualComposer: Main interface that replaces BimanualComposer

Usage:
    from modeling.ebm_compositionality import EBMBimanualComposer
    
    # Use exactly like original BimanualComposer
    model = EBMBimanualComposer(**config)
    model.load_pretrained_weights(left_path, right_path)
    
    # Inference (same interface)  
    actions = model.compute_trajectory(**observation)
"""

try:
    from .flow_to_energy import FlowToEnergyConverter
    from .energy_composer import EnergyComposer  
    from .ebm_bimanual_composer import EBMBimanualComposer
    
    __all__ = [
        'FlowToEnergyConverter',
        'EnergyComposer', 
        'EBMBimanualComposer'
    ]
    
except ImportError as e:
    print(f"Warning: Failed to import EBM components: {e}")
    print("EBM functionality may not be available.")
    __all__ = []

__version__ = "1.0.0"
