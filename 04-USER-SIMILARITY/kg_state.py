import numpy as np

class KGState(object):
    def __init__(self, embed_size, history_len=1):
        self.embed_size = embed_size
        self.history_len = history_len  # mode: one of {full, current}
        if history_len == 0:
            self.dim = 2 * embed_size
        elif history_len == 1:
            self.dim = 4 * embed_size
        elif history_len == 2:
            self.dim = 6 * embed_size
        else:
            raise Exception("history length should be one of {0, 1, 2}")

    def __call__(
        self,
        user_embed,
        node_embed,
        last_node_embed,
        last_relation_embed,
        older_node_embed,
        older_relation_embed,
    ):
        if self.history_len == 0:
            return np.concatenate([user_embed, node_embed])
        elif self.history_len == 1:
            return np.concatenate(
                [user_embed, node_embed, last_node_embed, last_relation_embed]
            )
        elif self.history_len == 2:
            return np.concatenate(
                [
                    user_embed,
                    node_embed,
                    last_node_embed,
                    last_relation_embed,
                    older_node_embed,
                    older_relation_embed,
                ]
            )
        else:
            raise Exception("mode should be one of {full, current}")