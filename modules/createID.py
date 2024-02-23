import random
import hashlib
import time
string = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstu"

def getID():
    result = hashlib.sha256(str(time.time()).encode("utf-8")).hexdigest()
    for i in range(6):
        result += string[random.randint(0,len(string)-1)]
    return result
