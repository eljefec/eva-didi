import logging
import mem_top

if __name__ == '__main__':
    logging.basicConfig(filename='logtest.log', level=logging.DEBUG)
    numbers = []
    for i in range(10):
        numbers.append(i)
        logging.debug(mem_top.mem_top())
