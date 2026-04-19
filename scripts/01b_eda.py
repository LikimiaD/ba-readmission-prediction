import logging

from src.data.eda import run_eda


def main():
    run_eda()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    main()
