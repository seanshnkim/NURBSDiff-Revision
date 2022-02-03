"""Defines the configuration to be loaded before running any experiment"""
from configobj import ConfigObj
import string


class Config(object):
    def __init__(self, filename: string):
        """
        Read from a config file
        :param filename: name of the file to read from
        """

        self.filename = filename
        config = ConfigObj(self.filename)
        self.config = config

        self.model_path = config["model"]["model_name"]
        self.pretrain_model_path = config["model"]["pretrain_model_path"]

        self.dataset_path = config["data"]["dataset"]
        self.proportion = config["data"].as_float("proportion")
        self.normals = config["data"].as_bool("normals")
        self.num_train = config["data"].as_int("num_train")
        self.num_val = config["data"].as_int("num_val")
        self.num_test = config["data"].as_int("num_test")

        self.epochs = config["train"].as_int("num_epochs")
        self.batch_size = config["train"].as_int("batch_size")
        self.num_point = config["train"].as_int("num_point")
        self.loss_weight = config["train"].as_float("loss_weight")

        # Learning rate
        self.optim = config["optimizer"]["optim"]
        self.lr = config["optimizer"].as_float("lr")
        self.lr_sch = config["optimizer"].as_bool("lr_sch")
        self.patience = config["optimizer"].as_int("patience")

        self.mode = config["network"].as_int("mode")
        self.grid_size = config["network"].as_int("grid_size")
        
        ## Added(2022.01.17)
        self.grid_size_list = list(map(int, config["network"].as_list("grid_size_list")))

        self.num_knot_layer = config["network"].as_int("num_knot_layer")
        self.num_cp_layer = config["network"].as_int("num_cp_layer")
        self.knot_channel = config["network"].as_int("num_knot_channel")
        self.cp_channel = config["network"].as_int("num_cp_channel")
        self.num_block = config["network"].as_int("num_block")
        self.dim_input = config["network"].as_int("dim_input")
        self.num_neighbor = config["network"].as_int("num_neighbor")
        self.dim_transformer = config["network"].as_int("dim_transformer")


    def write_config(self, filename):
        """
        Write the details of the experiment in the form of a config file.
        This will be used to keep track of what experiments are running and
        what parameters have been used.
        :return:
        """
        self.config.filename = filename
        self.config.write()

    def get_all_attribute(self):
        """
        This function prints all the values of the attributes, just to cross
        check whether all the data types are correct.
        :return: Nothing, just printing
        """
        for attr, value in self.__dict__.items():
            print(attr, value)


if __name__ == "__main__":
    file = Config("config_synthetic.yml")
    print(file.write_config())
