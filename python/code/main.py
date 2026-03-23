"""
Main testing script for Surface Data Processing Toolkit
"""

import sys
import os
import numpy as np

# Add the code directory to Python path so we can import our module
sys.path.append(os.path.join(os.path.dirname(__file__), 'code'))

from surface_io import SurfaceDataProcessor, load_surface_data, get_surface_data_info


def test_surface_io():
    """
    Test the surface data input/output functionality
    """
    print("=== Surface Data Processing Toolkit Test ===\n")

    # Initialize processor
    processor = SurfaceDataProcessor()

    # Test 1: Create dummy data for testing
    print("1. Testing with generated data...")
    try:
        # Create some dummy data to simulate real surface data
        dummy_left = np.random.normal(0, 1, 1000)  # 1000 vertices for left hemisphere
        dummy_right = np.random.normal(0, 1, 1000)  # 1000 vertices for right hemisphere

        # Test data info function
        info = processor.get_data_info(dummy_left)
        print(f"   Left hemisphere dummy data: {info['num_vertices']} vertices")
        print(f"   Data range: [{info['min']:.3f}, {info['max']:.3f}]")
        print("   ✓ Dummy data test passed\n")

    except Exception as e:
        print(f"   ✗ Dummy data test failed: {e}\n")

    # Test 2: Test file path handling
    print("2. Testing file path validation...")
    try:
        # This should raise FileNotFoundError
        processor.load_surface_data("nonexistent_file.gii")
        print("   ✗ File existence check failed")
    except FileNotFoundError:
        print("   ✓ File existence check passed")
    except Exception as e:
        print(f"   ✗ Unexpected error: {e}")

    try:
        # This should raise ValueError for unsupported format
        processor.load_surface_data("file.txt")
        print("   ✗ Format validation failed")
    except ValueError:
        print("   ✓ Format validation passed")
    except Exception as e:
        print(f"   ✗ Unexpected error: {e}")
    print()

    # Test 3: Test with actual data files (if they exist)
    print("3. Testing with actual data files...")

    # Define possible test file paths
    test_files = {
        'single_gii': os.path.join('../data', 'left_hemisphere.gii'),
        'left_gii': os.path.join('../data', 'left_hemisphere.gii'),
        'right_gii': os.path.join('../data', 'right_hemisphere.gii'),
        'mat_file': os.path.join('../data', 'example.mat')
    }

    # Test single .gii file
    if os.path.exists(test_files['single_gii']):
        try:
            data = processor.load_surface_data(test_files['single_gii'])
            info = processor.get_data_info(data)
            print(f"   ✓ Single .gii file loaded: {info['num_vertices']} vertices")
        except Exception as e:
            print(f"   ✗ Failed to load single .gii: {e}")
    else:
        print("   ⚠ Single .gii test file not found")

    # Test hemisphere pair
    if (os.path.exists(test_files['left_gii']) and
            os.path.exists(test_files['right_gii'])):
        try:
            data = processor.load_surface_data([
                test_files['left_gii'],
                test_files['right_gii']
            ])
            info = processor.get_data_info(data)
            print(f"   ✓ Hemisphere pair loaded: {info['num_vertices']} vertices")
        except Exception as e:
            print(f"   ✗ Failed to load hemisphere pair: {e}")
    else:
        print("   ⚠ Hemisphere pair test files not found")

    # Test .mat file
    if os.path.exists(test_files['mat_file']):
        try:
            # Try with auto-detection first, then with specific key if needed
            data = load_surface_data(test_files['mat_file'])
            info = get_surface_data_info(data)
            print(f"   ✓ .mat file loaded: {info['num_vertices']} vertices")
        except ValueError as e:
            if "specify data_key" in str(e):
                print("   ⚠ .mat file has multiple variables, need to specify data_key")
            else:
                print(f"   ✗ Failed to load .mat file: {e}")
        except Exception as e:
            print(f"   ✗ Failed to load .mat file: {e}")
    else:
        print("   ⚠ .mat test file not found")

    print()

    # Test 4: Test convenience functions
    print("4. Testing convenience functions...")
    try:
        # Create test data for convenience function test
        test_data = np.random.normal(0, 1, 500)

        # Test convenience functions
        info_conv = get_surface_data_info(test_data)
        print(f"   ✓ Convenience functions work: {info_conv['num_vertices']} vertices")
        print(f"   ✓ Data statistics - Mean: {info_conv['mean']:.3f}, Std: {info_conv['std']:.3f}")

    except Exception as e:
        print(f"   ✗ Convenience function test failed: {e}")

    print("\n=== Test Summary ===")
    print("All core functionality tests completed!")
    print("\nNext steps:")
    print("1. Add real .gii and .mat files to the data/ folder")
    print("2. Extend the toolkit with transformation functions")
    print("3. Add visualization capabilities")
    print("4. Integrate with neuromaps library")


def create_sample_data():
    """
    Create sample data files for testing if they don't exist
    """
    print("\n=== Creating Sample Data ===")

    data_dir = '../data'
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        print(f"Created directory: {data_dir}")

    # Create sample .mat file
    try:
        import scipy.io as sio
        sample_mat_data = {
            'cortical_thickness': np.random.normal(2.5, 0.5, 1000),
            'surface_area': np.random.normal(1.0, 0.2, 1000)
        }
        sio.savemat(os.path.join(data_dir, 'example.mat'), sample_mat_data)
        print("✓ Created sample .mat file")
    except Exception as e:
        print(f"✗ Could not create sample .mat file: {e}")

    print("Note: .gii files require specialized libraries to create.")
    print("Please use real neuroimaging data or install nibabel to create test .gii files.")


if __name__ == "__main__":
    # Create sample data first
    create_sample_data()

    # Run tests
    test_surface_io()