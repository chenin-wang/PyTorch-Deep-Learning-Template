import unittest
import io
import sys
import tempfile
import os
from loggers.logging_colors import get_logger
from transformers.utils import logging as hf_logging

# TODO : Add tests for the logger

class TestColoredLogger(unittest.TestCase):
    def setUp(self):
        self.original_stdout = sys.stdout
        self.captured_output = io.StringIO()
        sys.stdout = self.captured_output
        hf_logging.set_verbosity_debug()
        self.logger = get_logger()

    def tearDown(self):
        sys.stdout = self.original_stdout
        for handler in self.logger.handlers:
            handler.close()
            self.logger.removeHandler(handler)

    def test_logger_levels(self):
        test_messages = {
            'debug': 'This is a debug message',
            'info': 'This is an info message',
            'warning': 'This is a warning message',
            'error': 'This is an error message',
            'critical': 'This is a critical message'
        }
        
        for level, message in test_messages.items():
            with self.subTest(level=level):
                getattr(self.logger, level)(message)
                output = self.captured_output.getvalue()
                print(output)
                self.assertIn(message, output)
                self.assertIn(level.upper(), output)
                self.captured_output.truncate(0)
                self.captured_output.seek(0)

    def test_file_logging(self):
        with tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='.log') as temp_file:
            temp_log_path = temp_file.name
            file_logger = get_logger(temp_log_path)
            test_message = "Test file logging"
            file_logger.info(test_message)
            file_logger.debug("This is a debug message")

        try:
            with open(temp_log_path, 'r') as log_file:
                log_content = log_file.read()
                self.assertIn(test_message, log_content)
                self.assertIn("DEBUG", log_content)
        finally:
            for handler in file_logger.handlers:
                handler.close()
                file_logger.removeHandler(handler)
            os.unlink(temp_log_path)

if __name__ == '__main__':
    # python -m unittest test.test_log.py
    unittest.main()
