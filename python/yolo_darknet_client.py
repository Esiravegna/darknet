import subprocess
import json
from os.path import join


class DarkNetYOLOClient(object):
    """
    A python client for an already installed and compiled YOLO Network
    """

    def __init__(self,
                 path_binary='.',
                 path_weights='/mnt/hd/data/networks/yolo',
                 path_config='cfg/',
                 path_data='cfg/',
                 binary="darknet",
                 command="detector",
                 mode="test",
                 data="combine9k.data",
                 config="yolo9000.cfg",
                 weights="yolo9000.weights"
                 ):
        """
        The constructor

        :param path_binary (str): The path on where the darknet binary is located
        :param path_weights (str): The path on where the yolo9000.weights file is located
        :param path_config (str): The path on where the yolo9000.cfg is located
        :param path_data (str): The path on where the labels data is located
        :param binary (str): The binary, defaults to darknet
        :param command (str): The command to run. Defaults to detector
        :param mode (str): The mode to run the command, defaults to test
        :param data (str): The data label files, defaults to combine9k.data
        :param config (str): The config file (akin to network structure). Defaults to yolo9000.cfg
        :param weights (str): The weights file. Defaults to yolo9000.weights.
        """
        self.COMMAND = join(path_binary, command)
        self.WEIGHTS = join(path_weights, weights)
        self.DATA = join(path_data, data)
        self.CONFIG = join(path_config, config)
        self.COMMAND = command
        self.BINARY = join(path_binary, binary)
        self.MODE = mode
        self.THRESH = "-thersh {}"
        self.TRAVERSE_HIERARCHY = "-hier {}"

    def detect(self, image_file, threshold=0.25, hierarchy=0.5):
        """

        :param image_file (str): File to run trough the network
        :param threshold (float): The minimum threshold for returning a class. Defaults to 0.25
        :param hierarchy (float): The hierarchy trasverse confidence index. Defaults to 0.5
        :return: a Dict similar to:
            ```json
            {
               'detections':[
                  {
                     'left':69,
                     'bottom':117,
                     'prob':0.248203,
                     'right':99,
                     'top':78,
                     'class':'artifact'
                  },
                  {
                     'left':417,
                     'bottom':172,
                     'prob':0.739088,
                     'right':707,
                     'top':70,
                     'class':'truck'
                  },
                  {
                     'left':113,
                     'bottom':456,
                     'prob':0.268097,
                     'right':563,
                     'top':124,
                     'class':'bicycle'
                  },
                  {
                     'left':117,
                     'bottom':528,
                     'prob':0.812877,
                     'right':344,
                     'top':227,
                     'class':'feline'
                  }
               ]
            }

            ```
        """
        predictions = False
        yolo_command = [self.BINARY, self.COMMAND, self.MODE, self.DATA, self.CONFIG, self.WEIGHTS, image_file,
                        self.THRESH.format(threshold),
                        self.TRAVERSE_HIERARCHY.format(hierarchy)]

        yolo_process = subprocess.Popen(yolo_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = yolo_process.communicate()
        try:
            predictions = json.loads(out.decode('utf-8'))
        except (ValueError, TypeError) as e:
            raise Exception("Unable to process {} due to {} - {}".format(out, err, e))
        return predictions

