import logging
import logging.handlers

from fiber_photometry_analysis.utilities import colorize

LOGGING_NOTE = 25

config = {
    'logging': {
        'logging_level': "NOTE",
        'user_mail_address': ''
    },
    'system': {
        'admins_mail_addresses': [],
        'smtp_server': '',
        'smtp_port': '',
        'logger_mail_address': ''
    },
    'dest_dir': '/tmp/',
    'levels_colors': {
        'debug': 'white',
        'info': 'green',
        'note': 'yellow',
        'warning': 'yellow',
        'error': 'red',
        'critical': 'blink'
    }
}


class ColorFormatter(logging.Formatter):
    def format(self, record):
        base_str = super().format(record)
        formatted_str = colorize(base_str, config['levels_colors'][record.levelname.lower()])
        return formatted_str


def set_mail_handler(config, experiment_name, level, msg_base):
    dest_addresses = [config["logging"]["user_mail_address"]]
    if level in ('ERROR', 'CRITICAL'):
        dest_addresses.append(config["system"]["admins_mail_addresses"])
    handler = logging.handlers.SMTPHandler(
        mailhost=(config["system"]["smtp_server"],
                  config["system"]["smtp_port"]),
        fromaddr=config["system"]["logger_mail_address"],
        toaddrs=dest_addresses,
        subject="Fiber photometry experiment {} {}".format(experiment_name, msg_base))
    handler._timeout = 10.0  # Necessary for slow mail servers
    handler.setLevel(level)
    logging.getLogger('').addHandler(handler)


def set_color_print_handler():
    """

    :return:
    """
    formatter = ColorFormatter(fmt='%(levelname)s:%(message)s')
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    logging.getLogger('').addHandler(handler)


def init_logging(dest_dir, experiment_name, config):
    """
    Initialise the logging interface of the current experiment.
    """
    logging_level = config["logging"]["logging_level"]  # Here we put whether we debug, want info or only errors
    logging.addLevelName(LOGGING_NOTE, "NOTE")  # level between INFO and _WARNING
    # log_dest_path = os.path.join(dest_dir, "{}_fiber_photometry.log".format(experiment_name))
    set_color_print_handler()
    # EMAIL
    # send to user
    # set_mail_handler(config, experiment_name, 'NOTE', 'update')
    # send to admin
    set_mail_handler(config, experiment_name, 'ERROR', 'ERROR')  # check if ERROR or CRITICAL is best

    if logging_level == 'NOTE':
        logging_level_int = LOGGING_NOTE
    else:
        logging_level_int = getattr(logging, logging_level)
    logging.getLogger().setLevel(logging_level_int)


def test():
    init_logging(config['dest_dir'], 'test', config)
    logging.debug('A debug trace')
    logging.info('Just some info')
    logging.log(LOGGING_NOTE, msg='a note')
    logging.info('Some more info')
    logging.warning('a warning')
    logging.error('an error')


if __name__ == '__main__':
    test()
