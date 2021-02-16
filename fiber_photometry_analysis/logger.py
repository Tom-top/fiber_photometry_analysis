import os
import logging
import logging.handlers

LOGGING_NOTE = 25


def config_handler(config, experiment_name, level, msg_base):
    to_addresses = [config["logging"]["user_mail_address"]]
    if level in ('ERROR', 'CRITICAL'):
        to_addresses.append(config["system"]["admins_mail_addresses"])
    handler = logging.handlers.SMTPHandler(
        mailhost=(config["system"]["smtp_server"],
                  config["system"]["smtp_port"]),
        fromaddr=config["system"]["logger_mail_address"],
        toaddrs=to_addresses,
        subject="Fiber photometry experiment {} {}}".format(experiment_name, msg_base))
    handler._timeout = 10.0  # Necessary for slow mail servers
    handler.setLevel(level)
    logging.getLogger('').addHandler(handler)  # FIXME: also add print_in_colors with color as a function of level


def config_logging(dest_dir, experiment_name, config):
    """
    Initialise the logging interface of the current experiment.
    """
    logging_level = config["logging"]["logging_level"]  # Here we put whether we debug, want info or only errors
    log_dest = os.path.join(dest_dir, "{}_fiber_photometry.log".format(experiment_name))
    logging.addLevelName(LOGGING_NOTE, "NOTE")  # level between INFO and _WARNING
    logging.basicConfig(filename=log_dest, level=getattr(logging, logging_level))
    # send to user
    config_handler(config, experiment_name, 'NOTE', 'update')

    # send to admin
    config_handler(config, experiment_name, 'ERROR', 'ERROR')  # check if ERROR or CRITICAL is best


if __name__ == '__main__':
    logging.log(LOGGING_NOTE, msg='a note')
    logging.info('Just some info')
