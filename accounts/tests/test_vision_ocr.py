from django.test import SimpleTestCase

from accounts import vision_ocr


class NeedsVisionTest(SimpleTestCase):
    def test_empty_or_whitespace_needs_vision(self):
        self.assertTrue(vision_ocr.page_needs_vision(""))
        self.assertTrue(vision_ocr.page_needs_vision("   \n  "))

    def test_too_few_chars_needs_vision(self):
        self.assertTrue(vision_ocr.page_needs_vision("pg 12"))  # header-only scan

    def test_garbage_ratio_needs_vision(self):
        # A text layer that is mostly non-alphanumeric symbols = junk embedded OCR.
        self.assertTrue(vision_ocr.page_needs_vision("@#$%^&*<>{}|~`" * 20))

    def test_good_text_layer_skips_vision(self):
        good = ("Photosynthesis is the process by which green plants convert "
                "light energy into chemical energy stored in glucose. " * 3)
        self.assertFalse(vision_ocr.page_needs_vision(good))


class StripMarkersTest(SimpleTestCase):
    def test_strips_uncertainty_wrappers(self):
        self.assertEqual(
            vision_ocr.strip_uncertainty_markers("ATP is made in the [?thylakoid] membrane"),
            "ATP is made in the thylakoid membrane",
        )

    def test_leaves_plain_text_untouched(self):
        self.assertEqual(vision_ocr.strip_uncertainty_markers("no markers here"), "no markers here")


class ReconstructTest(SimpleTestCase):
    def test_reconstruct_returns_markdown_content(self):
        class _Msg:
            content = "## Photosynthesis\n- light reaction in [?thylakoid]"

        class _Choice:
            message = _Msg()

        class _Resp:
            choices = [_Choice()]

        class _FakeClient:
            class chat:
                class completions:
                    @staticmethod
                    def create(**kwargs):
                        return _Resp()

        vision_ocr._client = _FakeClient()  # bypass lazy builder
        try:
            md = vision_ocr.reconstruct_page_markdown(b"\x89PNG-bytes", page_number=1)
        finally:
            vision_ocr._client = None
        self.assertIn("## Photosynthesis", md)
        self.assertIn("[?thylakoid]", md)

    def test_reconstruct_raises_when_client_unavailable(self):
        vision_ocr._client = None
        vision_ocr._client_failed = True   # simulate "no key / import failed"
        try:
            with self.assertRaises(vision_ocr.VisionUnavailable):
                vision_ocr.reconstruct_page_markdown(b"x", page_number=1)
        finally:
            vision_ocr._client_failed = False
