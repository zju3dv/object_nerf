import torch
from torch import nn


class CodeLibrary(nn.Module):
    """
    Store various codes.
    """

    def __init__(self, model_config):
        super(CodeLibrary, self).__init__()

        self.embedding_instance = torch.nn.Embedding(
            model_config.get("N_max_objs", 64),
            model_config.get("N_obj_code_length", 64),
        )

    def forward(self, inputs):
        ret_dict = dict()

        # if 'frame_idx' in inputs:
        #     ret_dict['embedding_a'] = self.embedding_a(inputs['frame_idx'].squeeze())
        if "instance_ids" in inputs:
            ret_dict["embedding_instance"] = self.embedding_instance(
                inputs["instance_ids"].squeeze()
            )

        return ret_dict
