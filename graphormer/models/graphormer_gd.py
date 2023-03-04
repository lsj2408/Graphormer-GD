# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from fairseq import utils
from fairseq.models import (
    FairseqEncoder,
    FairseqEncoderModel,
    register_model,
    register_model_architecture,
)
from fairseq.modules import (
    LayerNorm,
)
from fairseq.utils import safe_hasattr

from ..modules import init_graphormer_params, GraphormerGraphEncoder, GraphormerGDGraphEncoder

logger = logging.getLogger(__name__)

from ..pretrain import load_pretrained_model


@register_model("graphormer_gd")
class GraphormerGDModel(FairseqEncoderModel):
    def __init__(self, args, encoder):
        super().__init__(encoder)
        self.args = args

        if getattr(args, "apply_graphormer_init", False):
            self.apply(init_graphormer_params)
        self.encoder_embed_dim = args.encoder_embed_dim
        if args.pretrained_model_name != "none":
            self.load_state_dict(load_pretrained_model(args.pretrained_model_name))
            if not args.load_pretrained_model_output_layer:
                self.encoder.reset_output_layer_parameters()

        if args.one_init_mul_bias:
            torch.nn.init.constant_(
                self.encoder.graph_encoder.graph_attn_bias_mul.spatial_encoder.weight, 1.0
            )
            torch.nn.init.constant_(
                self.encoder.graph_encoder.graph_attn_bias_mul.graph_token_virtual_distance.weight, 1.0
            )

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        # Arguments related to Graphormer-GD
        parser.add_argument(
            "--no-node-feature", action="store_true", help="do not use node feature"
        )

        parser.add_argument(
            "--no-edge-feature", action="store_true", help="do not use edge feature"
        )

        parser.add_argument(
            "--num-rd-bias-kernel", type=int, default=128, metavar="N", help="number of kernel function for resistance distance"
        )

        parser.add_argument(
            "--no-share-bias", action="store_true", help="do not share attention bias"
        )

        parser.add_argument(
            "--relu-mul-bias", action="store_true", help="use relu function for attn_bias_mul"
        )

        parser.add_argument(
            "--one-init-mul-bias", action="store_true", help="initiate attn_bias_mul by constant 1.0"
        )

        parser.add_argument(
            "--mul-bias-with-edge-feature", action="store_true", help="use edge feature in mul bias"
        )

        parser.add_argument(
            "--droppath-prob", type=float, default=0.0, metavar="D", help="droppath probability"
        )

        # Arguments related to dropout
        parser.add_argument(
            "--dropout", type=float, metavar="D", help="dropout probability"
        )
        parser.add_argument(
            "--attention-dropout",
            type=float,
            metavar="D",
            help="dropout probability for" " attention weights",
        )
        parser.add_argument(
            "--act-dropout",
            type=float,
            metavar="D",
            help="dropout probability after" " activation in FFN",
        )

        # Arguments related to hidden states and self-attention
        parser.add_argument(
            "--encoder-ffn-embed-dim",
            type=int,
            metavar="N",
            help="encoder embedding dimension for FFN",
        )
        parser.add_argument(
            "--encoder-layers", type=int, metavar="N", help="num encoder layers"
        )
        parser.add_argument(
            "--encoder-attention-heads",
            type=int,
            metavar="N",
            help="num encoder attention heads",
        )

        # Arguments related to input and output embeddings
        parser.add_argument(
            "--encoder-embed-dim",
            type=int,
            metavar="N",
            help="encoder embedding dimension",
        )
        parser.add_argument(
            "--share-encoder-input-output-embed",
            action="store_true",
            help="share encoder input" " and output embeddings",
        )
        parser.add_argument(
            "--encoder-learned-pos",
            action="store_true",
            help="use learned positional embeddings in the encoder",
        )
        parser.add_argument(
            "--no-token-positional-embeddings",
            action="store_true",
            help="if set, disables positional embeddings" " (outside self attention)",
        )
        parser.add_argument(
            "--max-positions", type=int, help="number of positional embeddings to learn"
        )

        # Arguments related to parameter initialization
        parser.add_argument(
            "--apply-graphormer-init",
            action="store_true",
            help="use custom param initialization for Graphormer",
        )

        # misc params
        parser.add_argument(
            "--activation-fn",
            choices=utils.get_available_activation_fns(),
            help="activation function to use",
        )
        parser.add_argument(
            "--encoder-normalize-before",
            action="store_true",
            help="apply layernorm before each encoder block",
        )
        parser.add_argument(
            "--pre-layernorm",
            action="store_true",
            help="apply layernorm before self-attention and ffn. Without this, post layernorm will used",
        )

    def max_nodes(self):
        return self.encoder.max_nodes

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""
        # make sure all arguments are present in older models
        base_architecture(args)

        if not safe_hasattr(args, "max_nodes"):
            args.max_nodes = args.tokens_per_sample

        logger.info(args)

        encoder = GraphormerGDEncoder(args)
        return cls(args, encoder)

    def forward(self, batched_data, **kwargs):
        return self.encoder(batched_data, **kwargs)


class GraphormerGDEncoder(FairseqEncoder):
    def __init__(self, args):
        super().__init__(dictionary=None)
        self.max_nodes = args.max_nodes

        self.graph_encoder = GraphormerGDGraphEncoder(
            # < for graphormer
            num_atoms=args.num_atoms,
            num_in_degree=args.num_in_degree,
            num_out_degree=args.num_out_degree,
            num_edges=args.num_edges,
            num_spatial=args.num_spatial,
            num_edge_dis=args.num_edge_dis,
            edge_type=args.edge_type,
            multi_hop_max_dist=args.multi_hop_max_dist,
            # >
            num_encoder_layers=args.encoder_layers,
            embedding_dim=args.encoder_embed_dim,
            ffn_embedding_dim=args.encoder_ffn_embed_dim,
            num_attention_heads=args.encoder_attention_heads,
            dropout=args.dropout,
            attention_dropout=args.attention_dropout,
            activation_dropout=args.act_dropout,
            encoder_normalize_before=args.encoder_normalize_before,
            pre_layernorm=args.pre_layernorm,
            apply_graphormer_init=args.apply_graphormer_init,
            activation_fn=args.activation_fn,

            # simulation
            no_node_feature=args.no_node_feature,
            no_edge_feature=args.no_edge_feature,
            num_rd_bias_kernel=args.num_rd_bias_kernel,
            no_share_bias=args.no_share_bias,
            relu_mul_bias=args.relu_mul_bias,
            droppath_prob=args.droppath_prob,
            mul_bias_with_edge_feature=args.mul_bias_with_edge_feature,
        )

        self.share_input_output_embed = args.share_encoder_input_output_embed
        self.embed_out = None
        self.lm_output_learned_bias = None

        # Remove head is set to true during fine-tuning
        self.load_softmax = not getattr(args, "remove_head", False)

        self.masked_lm_pooler = nn.Linear(
            args.encoder_embed_dim, args.encoder_embed_dim
        )

        self.lm_head_transform_weight = nn.Linear(
            args.encoder_embed_dim, args.encoder_embed_dim
        )
        self.activation_fn = utils.get_activation_fn(args.activation_fn)
        self.layer_norm = LayerNorm(args.encoder_embed_dim)

        self.lm_output_learned_bias = None
        if self.load_softmax:
            self.lm_output_learned_bias = nn.Parameter(torch.zeros(1))

            if not self.share_input_output_embed:
                self.embed_out = nn.Linear(
                    args.encoder_embed_dim, args.num_classes, bias=False
                )
            else:
                raise NotImplementedError

    def reset_output_layer_parameters(self):
        self.lm_output_learned_bias = nn.Parameter(torch.zeros(1))
        if self.embed_out is not None:
            self.embed_out.reset_parameters()

    def forward(self, batched_data, perturb=None, masked_tokens=None, **unused):
        inner_states, graph_rep = self.graph_encoder(
            batched_data,
            perturb=perturb,
        )

        x = inner_states[-1].transpose(0, 1)

        # project masked tokens only
        if masked_tokens is not None:
            raise NotImplementedError

        x = self.layer_norm(self.activation_fn(self.lm_head_transform_weight(x)))

        # project back to size of vocabulary
        if self.share_input_output_embed and hasattr(
            self.graph_encoder.embed_tokens, "weight"
        ):
            x = F.linear(x, self.graph_encoder.embed_tokens.weight)
        elif self.embed_out is not None:
            x = self.embed_out(x)
        if self.lm_output_learned_bias is not None:
            x = x + self.lm_output_learned_bias

        return x

    def max_nodes(self):
        """Maximum output length supported by the encoder."""
        return self.max_nodes

    def upgrade_state_dict_named(self, state_dict, name):
        if not self.load_softmax:
            for k in list(state_dict.keys()):
                if "embed_out.weight" in k or "lm_output_learned_bias" in k:
                    del state_dict[k]
        return state_dict


@register_model_architecture("graphormer_gd", "graphormer_gd")
def base_architecture(args):
    args.dropout = getattr(args, "dropout", 0.1)
    args.attention_dropout = getattr(args, "attention_dropout", 0.1)
    args.act_dropout = getattr(args, "act_dropout", 0.0)

    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 4096)
    args.encoder_layers = getattr(args, "encoder_layers", 6)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 8)

    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 1024)
    args.share_encoder_input_output_embed = getattr(
        args, "share_encoder_input_output_embed", False
    )
    args.no_token_positional_embeddings = getattr(
        args, "no_token_positional_embeddings", False
    )

    args.apply_graphormer_init = getattr(args, "apply_graphormer_init", False)

    args.activation_fn = getattr(args, "activation_fn", "gelu")
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", True)

    args.no_node_feature = getattr(args, "no_node_feature", False)
    args.no_edge_feature = getattr(args, "no_edge_feature", False)
    args.num_rd_bias_kernel = getattr(args, "num_rd_bias_kernel", 128)
    args.no_share_bias = getattr(args, "no_share_bias", False)
    args.relu_mul_bias = getattr(args, "relu_mul_bias", False)
    args.droppath_prob = getattr(args, "droppath_prob", 0.0)
    args.one_init_mul_bias = getattr(args, "one_init_mul_bias", False)
    args.mul_bias_with_edge_feature = getattr(args, "mul_bias_with_edge_feature", False)


@register_model_architecture("graphormer_gd", "graphormer_gd_base")
def graphormer_base_architecture(args):
    if args.pretrained_model_name == "pcqm4mv1_graphormer_base" or \
       args.pretrained_model_name == "pcqm4mv2_graphormer_base" or \
       args.pretrained_model_name == "pcqm4mv1_graphormer_base_for_molhiv":
        args.encoder_layers = 12
        args.encoder_attention_heads = 32
        args.encoder_ffn_embed_dim = 768
        args.encoder_embed_dim = 768
        args.dropout = getattr(args, "dropout", 0.0)
        args.attention_dropout = getattr(args, "attention_dropout", 0.1)
        args.act_dropout = getattr(args, "act_dropout", 0.1)
    else:
        args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 768)
        args.encoder_layers = getattr(args, "encoder_layers", 12)
        args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 32)
        args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 768)
        args.dropout = getattr(args, "dropout", 0.0)
        args.attention_dropout = getattr(args, "attention_dropout", 0.1)
        args.act_dropout = getattr(args, "act_dropout", 0.1)

    args.activation_fn = getattr(args, "activation_fn", "gelu")
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", True)
    args.apply_graphormer_init = getattr(args, "apply_graphormer_init", True)
    args.share_encoder_input_output_embed = getattr(
            args, "share_encoder_input_output_embed", False
        )
    args.no_token_positional_embeddings = getattr(
        args, "no_token_positional_embeddings", False
    )
    args.pre_layernorm = getattr(args, "pre_layernorm", False)
    base_architecture(args)


@register_model_architecture("graphormer_gd", "graphormer_gd_slim")
def graphormer_slim_architecture(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 80)

    args.encoder_layers = getattr(args, "encoder_layers", 12)

    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 8)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 80)

    args.activation_fn = getattr(args, "activation_fn", "gelu")
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", True)
    args.apply_graphormer_init = getattr(args, "apply_graphormer_init", True)
    args.share_encoder_input_output_embed = getattr(
            args, "share_encoder_input_output_embed", False
        )
    args.no_token_positional_embeddings = getattr(
        args, "no_token_positional_embeddings", False
    )
    args.pre_layernorm = getattr(args, "pre_layernorm", False)
    base_architecture(args)


@register_model_architecture("graphormer_gd", "graphormer_gd_large")
def graphormer_large_architecture(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 1024)

    args.encoder_layers = getattr(args, "encoder_layers", 24)

    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 32)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 1024)

    args.activation_fn = getattr(args, "activation_fn", "gelu")
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", True)
    args.apply_graphormer_init = getattr(args, "apply_graphormer_init", True)
    args.share_encoder_input_output_embed = getattr(
            args, "share_encoder_input_output_embed", False
        )
    args.no_token_positional_embeddings = getattr(
        args, "no_token_positional_embeddings", False
    )
    args.pre_layernorm = getattr(args, "pre_layernorm", False)
    base_architecture(args)
