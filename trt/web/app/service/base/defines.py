class Status:
    CREATED = 'CREATED'
    RESERVED = 'RESERVED'   # scheduled
    ALIVE = 'ALIVE'
    RECORDING = 'RECORDING'
    RUNNING = 'RUNNING'
    FAILED = 'FAILED'
    KILLED = 'KILLED'
    EXPIRED = 'EXPIRED'
    FINISHED = 'FINISHED'
    DONE = 'DONE'           # finished
    DELETED = 'DELETED'


class Topic:
    # GStreamer capture
    CAPTURE_FILTER = 'rtsp.capture.data'
    CAPTURE_STOP = 'rtsp.capture.stop'