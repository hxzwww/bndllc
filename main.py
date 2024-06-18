import sys
from detector import Detector

if __name__ == '__main__':
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    det = Detector()
    det.load(input_file)
    det.run()
    det.save(output_file)
