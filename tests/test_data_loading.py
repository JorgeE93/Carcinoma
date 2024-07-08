import unittest
from src.Carcinoma import DataFrame, DCMReader, Config


class TestDataFrame(unittest.TestCase):
    def setUp(self):
        self.data_frame = DataFrame()

    def test_get_vector(self):
        person, target = self.data_frame.get_vector()
        self.assertIsInstance(person, list)
        self.assertIsInstance(target, list)
        self.assertGreater(len(person), 0)
        self.assertGreater(len(target), 0)


class TestDCMReader(unittest.TestCase):
    def setUp(self):
        self.data_frame = DataFrame()
        self.dcm_reader = DCMReader(self.data_frame)

    def test_load_images(self):
        images = self.dcm_reader.load_images()
        self.assertIsInstance(images, list)
        self.assertGreater(len(images), 0)
        self.assertIsInstance(images[0], list)
        self.assertEqual(len(images[0]), 2)


class TestConfig(unittest.TestCase):
    def test_config(self):
        config = Config()
        self.assertEqual(config.epochs, 30)
        self.assertEqual(config.max_lr, 0.1)
        self.assertEqual(config.base_lr, 0.001)
        self.assertEqual(config.batch_size, 128)
        self.assertEqual(config.img_size, 512)


if __name__ == '__main__':
    unittest.main()
