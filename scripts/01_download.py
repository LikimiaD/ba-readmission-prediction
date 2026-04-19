import logging

from src.data.download import download_dataset


def main():
    dest = download_dataset()
    logging.getLogger('01_download').info('Raw data ready at %s', dest)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    main()
