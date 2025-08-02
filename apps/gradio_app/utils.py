import ffmpeg
import logging

def convert_to_supported_format(input_path, output_path):
    """Convert video to a browser-compatible format (MP4 with H.264 codec)."""
    try:
        stream = ffmpeg.input(input_path)
        stream = ffmpeg.output(stream, output_path, vcodec='h264', acodec='aac', format='mp4', loglevel='quiet')
        ffmpeg.run(stream)
        logging.debug(f"Converted video to {output_path}")
        return output_path
    except Exception as e:
        logging.error(f"Error converting video {input_path}: {str(e)}")
        return None