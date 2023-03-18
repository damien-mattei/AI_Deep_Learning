import logging

def create_logger(app_name=None):
    logger = logging.getLogger(app_name or __name__)
    logger.setLevel(logging.DEBUG)
    #log_format = '[%(asctime)-15s] [%(levelname)08s] %(funcName)s %(message)s'
    log_format = '%(message)s :: %(funcName)s'
    logging.basicConfig(format=log_format)
    return logger
