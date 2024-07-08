import unittest
import torch
import numpy as np
from PIL import Image
import tqdm as tqdm
import matplotlib.pyplot as plt
from src.Carcinoma import DataFrame, DCMReader


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

    def test_image(self):
        # Load and visualize images
        data_frame = self.data_frame
        dcm = DCMReader(data_frame)
        images = dcm.load_images()
        PIL_images = []
        for img, target in tqdm(images):
            im = Image.fromarray(np.uint16(img)).convert("L")
            PIL_images.append([im, target])

        fig, axs = plt.subplots(2, 5, figsize=(20, 8))
        for ax, (img, target) in zip(axs.flatten(), PIL_images[:10]):
            ax.imshow(img, cmap="gray")
            ax.set_title(f"Target: {target}")
            ax.axis("off")
        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    unittest.main()
