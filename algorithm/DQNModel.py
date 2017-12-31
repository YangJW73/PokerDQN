#!/bin/python

class DQNModel:

    def terminal_info_feat(self):
        raise NotImplementedError("Not implemented yet")

    def terminal_action_feat(self):
        raise NotImplementedError("Not implemented yet")

    def gen_info_feat(self, info):
        raise NotImplementedError("Not implemented yet")

    def gen_action_feat(self, action):
        raise NotImplementedError("Not implemented yet")

    def update_model(self, experiences):
        raise NotImplemented("Not implemented yet")

    def predict_q(self,info):
        raise NotImplementedError("Not implemented yet")

