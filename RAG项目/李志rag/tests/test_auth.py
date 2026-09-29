import unittest
from types import SimpleNamespace

from app.auth import create_token, hash_password, verify_password


class AuthTests(unittest.TestCase):
    def test_password_round_trip(self):
        encoded = hash_password("StrongPassword123!")
        self.assertTrue(verify_password("StrongPassword123!", encoded))
        self.assertFalse(verify_password("wrong-password", encoded))
        self.assertNotIn("StrongPassword123!", encoded)

    def test_token_created(self):
        user = SimpleNamespace(id=7, username="tester", is_admin=False)
        token = create_token(user)
        self.assertGreater(len(token), 30)


if __name__ == "__main__":
    unittest.main()

