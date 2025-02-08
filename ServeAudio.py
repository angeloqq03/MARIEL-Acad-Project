from flask import send_from_directory

def serve_audio(filename):
    return send_from_directory('audio', filename)