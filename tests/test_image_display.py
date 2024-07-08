import unittest
import torch
import matplotlib.pyplot as plt
from src.Carcinoma import DataFrame


class TestImageDisplay(unittest.TestCase):
    def setUp(self):
        self.data_frame = DataFrame()

    def test_show_grid(self):
        # Create a sample image tensor
        sample_image = torch.rand((3, 512, 512))  # Simulate a random image tensor

        try:
            self.data_frame.show_grid(sample_image, title="Sample Image")
            plt.close('all')  # Close the plot to avoid display during testing
        except Exception as e:
            self.fail(f"show_grid() raised an exception: {e}")


if __name__ == '__main__':
    unittest.main()
