"""
Surface Data Input/Output Module
Handles reading of neuroimaging surface data files (.gii, .mat)
"""

import numpy as np
import nibabel as nib
from scipy.io import loadmat
import os
from typing import Union, List, Optional


class SurfaceDataProcessor:
    """
    Surface Brain Space Data Processing Toolkit
    Functionality similar to neuromaps for processing and comparing surface brain space data
    """

    def __init__(self):
        self.supported_formats = ['.gii', '.mat']

    def load_surface_data(self, file_path: Union[str, List[str]],
                          data_key: Optional[str] = None) -> np.ndarray:
        """
        Load surface data files, supporting .gii and .mat formats

        Parameters:
        ----------
        file_path : str or list of str
            File path(s). Can be a single file path or list of left/right hemisphere file paths
        data_key : str, optional
            For .mat files, specify the data key to read. If None, auto-detection is attempted

        Returns:
        -------
        np.ndarray
            Vector containing values for each vertex. For single hemisphere, returns as is;
            for left/right hemispheres, returns concatenated data

        Raises:
        -------
        FileNotFoundError
            When specified file does not exist
        ValueError
            When file format is unsupported or data reading fails
        """

        # Handle single file or file list
        if isinstance(file_path, str):
            return self._load_single_file(file_path, data_key)
        elif isinstance(file_path, list):
            if len(file_path) == 1:
                return self._load_single_file(file_path[0], data_key)
            elif len(file_path) == 2:
                return self._load_hemisphere_data(file_path, data_key)
            else:
                raise ValueError("Maximum 2 files supported (left/right hemispheres)")
        else:
            raise TypeError("file_path must be a string or list of strings")

    def _load_single_file(self, file_path: str, data_key: Optional[str] = None) -> np.ndarray:
        """
        Load a single surface data file
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        _, ext = os.path.splitext(file_path)
        ext = ext.lower()

        if ext not in self.supported_formats:
            raise ValueError(f"Unsupported file format: {ext}. Supported formats: {self.supported_formats}")

        if ext == '.gii':
            return self._load_gii_file(file_path)
        elif ext == '.mat':
            return self._load_mat_file(file_path, data_key)

    def _load_gii_file(self, file_path: str) -> np.ndarray:
        """
        Load Gifti format file (.gii)
        """
        try:
            gii_data = nib.load(file_path)

            if len(gii_data.darrays) == 0:
                raise ValueError(f"Gifti file contains no data: {file_path}")

            data_array = gii_data.darrays[0].data

            if data_array.ndim > 1:
                data_array = data_array.flatten()

            return data_array.astype(np.float64)

        except Exception as e:
            raise ValueError(f"Failed to read Gifti file: {file_path}. Error: {str(e)}")

    def _load_mat_file(self, file_path: str, data_key: Optional[str] = None) -> np.ndarray:
        """
        Load MATLAB format file (.mat)
        """
        try:
            mat_data = loadmat(file_path)

            # Remove MATLAB metadata keys
            mat_data_clean = {key: value for key, value in mat_data.items()
                              if not key.startswith('__')}

            if not mat_data_clean:
                raise ValueError(f"MAT file contains no valid data: {file_path}")

            # Determine which data key to use
            if data_key is not None:
                if data_key not in mat_data_clean:
                    available_keys = list(mat_data_clean.keys())
                    raise KeyError(f"Key '{data_key}' not found. Available keys: {available_keys}")
                target_key = data_key
            else:
                # Auto-select first key (if only one data variable)
                if len(mat_data_clean) == 1:
                    target_key = list(mat_data_clean.keys())[0]
                else:
                    available_keys = list(mat_data_clean.keys())
                    raise ValueError(
                        f"MAT file contains multiple data variables, please specify data_key. Available keys: {available_keys}"
                    )

            data_array = mat_data_clean[target_key]

            if data_array.ndim > 1:
                data_array = data_array.flatten()

            return data_array.astype(np.float64)

        except Exception as e:
            raise ValueError(f"Failed to read MAT file: {file_path}. Error: {str(e)}")

    def _load_hemisphere_data(self, file_paths: List[str],
                              data_key: Optional[str] = None) -> np.ndarray:
        """
        Load and merge left/right hemisphere data
        """
        if len(file_paths) != 2:
            raise ValueError("Exactly 2 file paths required for left/right hemisphere data")

        left_data = self._load_single_file(file_paths[0], data_key)
        right_data = self._load_single_file(file_paths[1], data_key)

        combined_data = np.concatenate([left_data, right_data])
        return combined_data

    def get_data_info(self, data: np.ndarray) -> dict:
        """
        Get basic information about surface data
        """
        return {
            'shape': data.shape,
            'min': np.min(data),
            'max': np.max(data),
            'mean': np.mean(data),
            'std': np.std(data),
            'num_vertices': len(data),
            'has_nan': np.any(np.isnan(data)),
            'has_inf': np.any(np.isinf(data))
        }


# Convenience functions
def load_surface_data(file_path: Union[str, List[str]],
                      data_key: Optional[str] = None) -> np.ndarray:
    """
    Convenience function: Load surface data files
    """
    processor = SurfaceDataProcessor()
    return processor.load_surface_data(file_path, data_key)


def get_surface_data_info(data: np.ndarray) -> dict:
    """
    Convenience function: Get basic information about surface data
    """
    processor = SurfaceDataProcessor()
    return processor.get_data_info(data)