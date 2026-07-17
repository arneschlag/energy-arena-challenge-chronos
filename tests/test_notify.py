from __future__ import annotations

import unittest

from pipeline.notify import NotificationConfigError, SMTPConfig, send_mail


class FakeSMTP:
    instances = []

    def __init__(self, host, port, timeout):
        self.host = host
        self.port = port
        self.timeout = timeout
        self.started_tls = False
        self.login_args = None
        self.message = None
        self.__class__.instances.append(self)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return None

    def starttls(self):
        self.started_tls = True

    def login(self, username, password):
        self.login_args = (username, password)

    def send_message(self, message):
        self.message = message


class NotifyTest(unittest.TestCase):
    def setUp(self):
        FakeSMTP.instances.clear()

    def test_smtp_aliases_and_send(self):
        env = {
            "ALERT_SMTP_HOST": "mail.example.test",
            "ALERT_SMTP_PORT": "587",
            "SMTP_USER": "robot@example.test",
            "SMTP_PASSWORD": "secret-password",
            "ALERT_EMAIL_TO": "one@example.test; two@example.test",
        }
        config = SMTPConfig.from_env(env)
        send_mail("Subject", "Body", config=config, smtp_factory=FakeSMTP)
        smtp = FakeSMTP.instances[0]
        self.assertTrue(smtp.started_tls)
        self.assertEqual(smtp.login_args, ("robot@example.test", "secret-password"))
        self.assertEqual(smtp.message["To"], "one@example.test, two@example.test")
        self.assertEqual(smtp.message.get_content().strip(), "Body")

    def test_missing_configuration_error_does_not_contain_password(self):
        password = "do-not-print-me"
        with self.assertRaises(NotificationConfigError) as caught:
            SMTPConfig.from_env({"SMTP_PASSWORD": password})
        self.assertNotIn(password, str(caught.exception))


if __name__ == "__main__":
    unittest.main()
