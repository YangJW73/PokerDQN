#!/bin/python

class Experience:
    def __init__(self, turn, info_feat, action_feat, reward, next_info_feat, next_available_action_feats):
        self.turn                        = turn
        self.info_feat                   = info_feat
        self.action_feat                 = action_feat
        self.reward                      = reward
        self.next_info_feat              = next_info_feat
        self.next_available_action_feats = next_available_action_feats