#!/usr/bin/env python3
from getpass import getpass
from werkzeug.security import generate_password_hash


password = getpass("Password: ")
confirm = getpass("Confirm: ")
if password != confirm:
    raise SystemExit("Passwords do not match")
print(generate_password_hash(password, method="scrypt"))
