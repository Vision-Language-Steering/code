import torch
import numpy as np


def get_device_from_parameters(model):
    """Get device from model parameters"""
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device('cpu')


def get_dtype_from_parameters(model):
    """Get dtype from model parameters"""
    try:
        return next(model.parameters()).dtype
    except StopIteration:
        return torch.float32


def merge_dicts(dicts):
    return {
        k : v 
        for d in dicts
        for k, v in d.items()
    }
    
def exec_safe(code_str, gvars=None, lvars=None):
    # Allow import statements for VLM-generated guidance functions
    banned_phrases = ['__']  # Only ban dunder methods for safety
    for phrase in banned_phrases:
        assert phrase not in code_str
  
    if gvars is None:
        gvars = {}
    if lvars is None:
        lvars = {}
    empty_fn = lambda *args, **kwargs: None
    custom_gvars = merge_dicts([
        gvars,
        {'exec': empty_fn, 'eval': empty_fn}
    ])
    try:
        exec(code_str, custom_gvars, lvars)
    except Exception as e:
        print(f'Error executing code:\n{code_str}')
        raise e

def load_functions_from_txt(txt_path, validate=True):
    """
    Load guidance functions from text file.
    
    Note: The generated functions should use torch instead of numpy,
    because the input trajectory_3d and keypoints are torch.Tensor
    
    Args:
        txt_path: Path to the guidance function text file
        validate: If True, validate functions with dummy inputs before returning
        
    Returns:
        List of wrapped guidance functions
        
    Raises:
        Exception: If validation fails and validate=True
    """
    if txt_path is None:
        return []
    # load txt file
    with open(txt_path, 'r') as f:
        functions_text = f.read()
    
    # Replace numpy with torch in the code to ensure compatibility
    # This is a simple fix for VLM-generated code that might use numpy
    # Replace common numpy operations with torch equivalents
    import re
    
    # Replace numpy array creation
    functions_text = functions_text.replace('np.array(', 'torch.tensor(')
    functions_text = functions_text.replace('numpy.array(', 'torch.tensor(')
    
    # Replace numpy operations with torch equivalents
    replacements = {
        'np.zeros': 'torch.zeros',
        'np.ones': 'torch.ones',
        'np.linalg.norm': 'torch.norm',
        'np.sum': 'torch.sum',
        'np.mean': 'torch.mean',
        'np.max': 'torch.max',
        'np.min': 'torch.min',
        'np.sqrt': 'torch.sqrt',
        'np.abs': 'torch.abs',
        'np.exp': 'torch.exp',
        'np.log': 'torch.log',
    }
    
    for np_func, torch_func in replacements.items():
        functions_text = functions_text.replace(np_func, torch_func)
    
    # execute functions
    gvars_dict = {
        'torch': torch,
    }  # external library APIs
    lvars_dict = dict()
    exec_safe(functions_text, gvars=gvars_dict, lvars=lvars_dict)
    
    # Wrap all callable functions to ensure device consistency
    wrapped_functions = []
    for name, func in lvars_dict.items():
        if callable(func):
            wrapped_functions.append(wrap_guidance_function_with_device_fix(func))
    
    # Validate functions with dummy inputs
    if validate and wrapped_functions:
        _validate_guidance_functions(wrapped_functions, txt_path)
    
    return wrapped_functions


def _validate_guidance_functions(functions, source_path="unknown"):
    """
    Validate guidance functions with dummy inputs to catch errors early.
    
    Args:
        functions: List of guidance functions to validate
        source_path: Path to source file (for error messages)
        
    Raises:
        ValueError: If any function fails validation
    """
    # Create dummy inputs matching expected shapes
    # keypoints: (N, 3), trajectory: (B, T, 3)
    # Use 30 keypoints to cover typical LIBERO scenes (usually 15-25 keypoints)
    dummy_keypoints = torch.randn(30, 3)  # 30 keypoints to handle most scenes
    dummy_trajectory = torch.randn(1, 10, 3)  # batch=1, 10 timesteps, 3D positions
    
    for i, func in enumerate(functions):
        try:
            result = func(dummy_keypoints, dummy_trajectory)
            
            # Check result is valid
            if result is None:
                raise ValueError(f"Function {i} returned None")
            
            if not isinstance(result, torch.Tensor):
                raise ValueError(f"Function {i} returned {type(result)}, expected torch.Tensor")
            
            # Check result requires grad (needed for guidance)
            if not result.requires_grad:
                # This is a warning, not an error - some functions may not need grad
                print(f"Warning: Function {i} from {source_path} returned tensor without grad")
                
        except Exception as e:
            raise ValueError(
                f"Guidance function validation failed for {source_path} (function {i}): {e}\n"
                f"This may be due to VLM generating incorrect code. "
                f"Check the guidance function file for errors."
            ) from e


def wrap_guidance_function_with_device_fix(func):
    """
    Wrap guidance function, automatically fix all device issues created by tensors.
    
    By creating a temporary torch module wrapper, all tensors created in the function
    will automatically use the device of trajectory_3d.
    """
    def wrapped(keypoints, trajectory_3d):
        """
        Args:
            trajectory_3d: (B, horizon+1, 3) tensor
            keypoints: (N, 3) tensor
        """
        # Get device from input
        device = trajectory_3d.device
        dtype = trajectory_3d.dtype
        
        # Ensure keypoints on same device
        if keypoints.device != device:
            keypoints = keypoints.to(device)
        
        # Create a custom torch module that makes all tensors on the correct device
        class DeviceAwareTorchModule:
            """Wrapper around torch that ensures all operations use the correct device"""
            
            def __init__(self, target_device, target_dtype):
                self._device = target_device
                self._dtype = target_dtype
                self._original_torch = torch
            
            def tensor(self, *args, **kwargs):
                if 'device' not in kwargs:
                    kwargs['device'] = self._device
                if 'dtype' not in kwargs and len(args) > 0:
                    # Try to infer if it's float data
                    try:
                        first_arg = args[0]
                        if isinstance(first_arg, (list, tuple)) and len(first_arg) > 0:
                            if isinstance(first_arg[0], (float, int)):
                                kwargs['dtype'] = self._dtype
                    except:
                        pass
                return self._original_torch.tensor(*args, **kwargs)
            
            def zeros(self, *args, **kwargs):
                if 'device' not in kwargs:
                    kwargs['device'] = self._device
                return self._original_torch.zeros(*args, **kwargs)
            
            def ones(self, *args, **kwargs):
                if 'device' not in kwargs:
                    kwargs['device'] = self._device
                return self._original_torch.ones(*args, **kwargs)
            
            def randn(self, *args, **kwargs):
                if 'device' not in kwargs:
                    kwargs['device'] = self._device
                return self._original_torch.randn(*args, **kwargs)
            
            def rand(self, *args, **kwargs):
                if 'device' not in kwargs:
                    kwargs['device'] = self._device
                return self._original_torch.rand(*args, **kwargs)
            
            def __getattr__(self, name):
                # For all other attributes, delegate to original torch
                return getattr(self._original_torch, name)
        
        # Monkey-patch torch in the function's globals
        if hasattr(func, '__globals__'):
            original_torch = func.__globals__.get('torch', torch)
            device_aware_torch = DeviceAwareTorchModule(device, dtype)
            
            # Temporarily replace torch
            func.__globals__['torch'] = device_aware_torch
            
            try:
                result = func(keypoints, trajectory_3d)
                
                # Ensure result is on correct device
                if result is not None and hasattr(result, 'device') and result.device != device:
                    result = result.to(device)
                
                return result
            finally:
                # Restore original torch
                func.__globals__['torch'] = original_torch
        else:
            # Fallback: just call the function
            return func(keypoints, trajectory_3d)
    
    return wrapped


def wrap_guidance_function(func):
    """
    Wrap guidance function to ensure all operations are on the correct device.
    
    Args:
        func: original guidance function
    
    Returns:
        wrapped_func: wrapped function, automatically handle device issues
    """
    def wrapped(trajectory_3d, keypoints):
        """
        Args:
            trajectory_3d: (B, horizon+1, 3) tensor
            keypoints: (N, 3) tensor
        """
        # Get device from input tensors
        device = trajectory_3d.device
        dtype = trajectory_3d.dtype
        
        # Ensure keypoints on same device
        if keypoints.device != device:
            keypoints = keypoints.to(device)
        
        # Call original function
        try:
            result = func(keypoints, trajectory_3d)
            
            # Ensure result is on correct device
            if result is not None and hasattr(result, 'device'):
                if result.device != device:
                    result = result.to(device)
            
            return result
            
        except RuntimeError as e:
            if "device" in str(e).lower():
                # Try to fix device issues by moving everything to trajectory_3d's device
                print(f"Device mismatch detected, attempting to fix...")
                
                # This is a fallback - wrap torch operations
                import sys
                original_tensor = torch.tensor
                
                def device_aware_tensor(*args, **kwargs):
                    if 'device' not in kwargs:
                        kwargs['device'] = device
                    if 'dtype' not in kwargs and len(args) > 0:
                        kwargs['dtype'] = dtype
                    return original_tensor(*args, **kwargs)
                
                # Temporarily replace torch.tensor
                torch.tensor = device_aware_tensor
                try:
                    result = func(keypoints, trajectory_3d)
                    if result is not None and hasattr(result, 'device') and result.device != device:
                        result = result.to(device)
                    return result
                finally:
                    # Restore original
                    torch.tensor = original_tensor
            else:
                raise
    
    return wrapped