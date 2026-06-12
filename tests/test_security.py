import unittest
from rag.utils.security import validate_filename, validate_file_content
from rag.utils.exceptions import SecurityError


class TestSecurity(unittest.TestCase):
    def test_valid_filenames(self):
        self.assertEqual(validate_filename("report.pdf"), "report.pdf")
        self.assertEqual(validate_filename("data_file.txt"), "data_file.txt")
        self.assertEqual(validate_filename("spaces in name.md"), "spaces_in_name.md")

    def test_invalid_filenames_traversal(self):
        with self.assertRaises(SecurityError):
            validate_filename("../etc/passwd.txt")
            
        with self.assertRaises(SecurityError):
            validate_filename("some_folder/../../report.pdf")

    def test_missing_extension(self):
        with self.assertRaises(SecurityError):
            validate_filename("no_extension_file")

    def test_valid_pdf_magic_bytes(self):
        # PDFs must start with %PDF
        content = b"%PDF-1.4\n1 0 obj..."
        self.assertTrue(validate_file_content(content, "document.pdf"))

    def test_invalid_pdf_magic_bytes(self):
        content = b"Not a PDF file structure..."
        with self.assertRaises(SecurityError):
            validate_file_content(content, "document.pdf")

    def test_valid_text_utf8(self):
        content = "Hello, world!".encode("utf-8")
        self.assertTrue(validate_file_content(content, "notes.txt"))

    def test_invalid_text_null_bytes(self):
        # Plain text should not contain binary null bytes
        content = b"Some text\x00with null bytes"
        with self.assertRaises(SecurityError):
            validate_file_content(content, "notes.txt")


if __name__ == "__main__":
    unittest.main()
