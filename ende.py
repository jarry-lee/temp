import os
import json
from cryptography.fernet import Fernet

def get_encryption_key(models_dir):
    key_path = os.path.join(models_dir, ".secret.key")
    if not os.path.exists(key_path):
        key = Fernet.generate_key()
        with open(key_path, "wb") as f:
            f.write(key)
    else:
        with open(key_path, "rb") as f:
            key = f.read()
    return key

def encrypt_and_save(data, file_path, models_dir):
    fernet = Fernet(get_encryption_key(models_dir))
    encrypted = fernet.encrypt(json.dumps(data, indent=4).encode('utf-8'))
    with open(file_path, "wb") as f:
        f.write(encrypted)

def load_and_decrypt(file_path, models_dir):
    if not os.path.exists(file_path):
        return []
        
    fernet = Fernet(get_encryption_key(models_dir))
    try:
        with open(file_path, "rb") as f:
            decrypted = fernet.decrypt(f.read())
            return json.loads(decrypted)
    except Exception:
        # Fallback for unencrypted legacy JSON
        try:
            with open(file_path, "r") as f:
                return json.load(f)
        except:
            return []
