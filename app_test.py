import unittest

from main import convert_io_string_to_image_rgb

class MainTest(unittest.TestCase):

    def test_convert_io_string_to_image(self):
        file = open("./test_sample/005.png", "r")
        file_rgb = convert_io_string_to_image_rgb(file)

        self.assertEqual(file.shape, file_rgb.shape)

if __name__ == '__main__':
    unittest.main()