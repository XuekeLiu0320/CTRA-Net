# model.py
# -*- coding: utf-8 -*-
"""
CTRA-Net main model
Corresponding to the paper:
Cross-Temporal Relational Alignment for Unified Knowledge Tracing

This file defines the overall model pipeline:
1. Interaction embedding
2. Temporal encoding by GRU
3. Cross-Temporal Graph Attention Propagation (CT-GAP)
4. Temporal-Relational Memory Bank (TRMB)
5. Fusion and next-step prediction
6. Joint loss computation

Expected external files:
- config.py
- modules.py
- utils.py
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from modules import (
    InteractionEmbedding,
    CrossTemporalGraphAttentionPropagation,
    TemporalRelationalMemoryBank,
)


class CTRANet(nn.Module):
    """
    Cross-Temporal Relational Alignment Network (CTRA-Net)

    Input:
        question_ids: LongTensor, shape [B, T]
        responses: LongTensor / FloatTensor, shape [B, T], values in {0,1}
        mask: FloatTensor / BoolTensor, shape [B, T], 1 means valid step

    Output:
        A dict containing:
            - logits: [B, T-1]
            - probs: [B, T-1]
            - y_true: [B, T-1]
            - valid_mask: [B, T-1]
            - loss_dict: dict of losses
    """

    def __init__(
        self,
        num_questions: int,
        num_concepts: int,
        hidden_dim: int = 128,
        embed_dim: int = 128,
        node_dim: int = 128,
        memory_size: int = 64,
        dropout: float = 0.2,
        tau: float = 0.5,
        gamma: float = 0.01,
        memory_write_step: float = 0.5,
        lambda_align: float = 0.1,
        lambda_entropy: float = 0.01,
        adjacency_matrix: Optional[torch.Tensor] = None,
        question_concept_map: Optional[torch.Tensor] = None,
        use_concept_embedding: bool = True,
    ) -> None:
        super().__init__()

        self.num_questions = num_questions
        self.num_concepts = num_concepts
        self.hidden_dim = hidden_dim
        self.embed_dim = embed_dim
        self.node_dim = node_dim
        self.memory_size = memory_size
        self.tau = tau
        self.gamma = gamma
        self.memory_write_step = memory_write_step
        self.lambda_align = lambda_align
        self.lambda_entropy = lambda_entropy
        self.use_concept_embedding = use_concept_embedding

        # -------------------------------------------------
        # 1) Interaction Embedding
        # -------------------------------------------------
        self.interaction_embed = InteractionEmbedding(
            num_questions=num_questions,
            embed_dim=embed_dim,
            use_response_embedding=True,
        )

        # -------------------------------------------------
        # 2) Temporal Encoder
        # -------------------------------------------------
        self.gru = nn.GRU(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
        )

        self.temporal_dropout = nn.Dropout(dropout)

        # -------------------------------------------------
        # 3) Structural Node Representation
        # -------------------------------------------------
        # Concept/node embeddings used by CT-GAP
        self.node_embedding = nn.Embedding(num_concepts, node_dim)

        # Optional question embedding for prediction
        self.question_embedding = nn.Embedding(num_questions, hidden_dim)

        # Register adjacency matrix
        if adjacency_matrix is None:
            adjacency_matrix = torch.eye(num_concepts, dtype=torch.float32)
        self.register_buffer("adjacency_matrix", adjacency_matrix.float())

        # Register question -> concept mapping
        # shape [num_questions], each value in [0, num_concepts-1]
        if question_concept_map is None:
            question_concept_map = torch.zeros(num_questions, dtype=torch.long)
        self.register_buffer("question_concept_map", question_concept_map.long())

        # -------------------------------------------------
        # 4) CT-GAP
        # -------------------------------------------------
        self.ct_gap = CrossTemporalGraphAttentionPropagation(
            hidden_dim=hidden_dim,
            node_dim=node_dim,
            dropout=dropout,
        )

        # Project relation modulation vector to memory space
        self.rel_proj = nn.Linear(node_dim, hidden_dim)

        # -------------------------------------------------
        # 5) TRMB
        # -------------------------------------------------
        self.trmb = TemporalRelationalMemoryBank(
            memory_size=memory_size,
            memory_dim=hidden_dim,
            tau=tau,
            gamma=gamma,
            write_step=memory_write_step,
        )

        # -------------------------------------------------
        # 6) Fusion & Prediction
        # -------------------------------------------------
        fusion_dim = hidden_dim * 4
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.predictor = nn.Linear(hidden_dim, 1)

        self.layer_norm = nn.LayerNorm(hidden_dim)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.node_embedding.weight)
        nn.init.xavier_uniform_(self.question_embedding.weight)

        for name, param in self.gru.named_parameters():
            if "weight" in name:
                nn.init.xavier_uniform_(param)
            elif "bias" in name:
                nn.init.zeros_(param)

        for module in self.fusion:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

        nn.init.xavier_uniform_(self.predictor.weight)
        nn.init.zeros_(self.predictor.bias)
        nn.init.xavier_uniform_(self.rel_proj.weight)
        nn.init.zeros_(self.rel_proj.bias)

    # ------------------------------------------------------------------
    # helper functions
    # ------------------------------------------------------------------
    def get_question_concepts(self, question_ids: torch.Tensor) -> torch.Tensor:
        """
        Map question IDs to concept IDs.
        question_ids: [B, T]
        return: concept_ids [B, T]
        """
        return self.question_concept_map[question_ids]

    def gather_target_node_state(
        self,
        struct_nodes: torch.Tensor,
        target_concept_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Gather target concept representation from propagated node matrix.

        struct_nodes: [B, N, D]
        target_concept_ids: [B]
        return: [B, D]
        """
        batch_size = struct_nodes.size(0)
        batch_index = torch.arange(batch_size, device=struct_nodes.device)
        return struct_nodes[batch_index, target_concept_ids, :]

    @staticmethod
    def masked_bce_loss(
        logits: torch.Tensor,
        targets: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        logits: [B, L]
        targets: [B, L]
        mask: [B, L]
        """
        bce = F.binary_cross_entropy_with_logits(
            logits,
            targets.float(),
            reduction="none",
        )
        loss = (bce * mask.float()).sum() / (mask.float().sum().clamp_min(1.0))
        return loss

    @staticmethod
    def masked_mse_loss(
        x: torch.Tensor,
        y: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        x, y: [B, L, D]
        mask: [B, L]
        """
        mse = ((x - y) ** 2).sum(dim=-1)
        loss = (mse * mask.float()).sum() / (mask.float().sum().clamp_min(1.0))
        return loss

    @staticmethod
    def masked_entropy_loss(
        attn_prob: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        attn_prob: [B, L, M]
        mask: [B, L]
        """
        entropy = -(attn_prob.clamp_min(1e-12) * attn_prob.clamp_min(1e-12).log()).sum(dim=-1)
        loss = (entropy * mask.float()).sum() / (mask.float().sum().clamp_min(1.0))
        return loss

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------
    def forward(
        self,
        question_ids: torch.Tensor,
        responses: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_details: bool = True,
    ) -> Dict[str, Any]:
        """
        Forward for training and evaluation.

        The prediction target is next-step correctness:
            use information at time t to predict response at time t+1

        Args:
            question_ids: [B, T]
            responses: [B, T]
            mask: [B, T], optional
            return_details: whether return intermediate values

        Returns:
            dict
        """
        device = question_ids.device
        batch_size, seq_len = question_ids.size()

        if mask is None:
            mask = torch.ones_like(question_ids, dtype=torch.float32, device=device)
        else:
            mask = mask.float()

        # --------------------------------------------
        # 1) interaction embedding
        # --------------------------------------------
        interaction_emb = self.interaction_embed(question_ids, responses)  # [B, T, E]

        # --------------------------------------------
        # 2) temporal encoding
        # --------------------------------------------
        temporal_states, _ = self.gru(interaction_emb)  # [B, T, H]
        temporal_states = self.temporal_dropout(temporal_states)

        # --------------------------------------------
        # 3) initialize memory for each batch
        # --------------------------------------------
        self.trmb.reset_state(batch_size=batch_size, device=device)

        # static node embeddings
        node_repr = self.node_embedding.weight  # [N, D]
        adj = self.adjacency_matrix  # [N, N]

        # collect step-wise outputs
        logits_list = []
        probs_list = []
        y_true_list = []
        valid_mask_list = []

        rel_vector_list = []
        memory_context_list = []
        memory_prob_list = []

        # predict y_{t+1} using state at t
        for t in range(seq_len - 1):
            h_t = temporal_states[:, t, :]  # [B, H]

            # ----------------------------------------
            # 3.1 CT-GAP
            # ----------------------------------------
            # struct_nodes_t: [B, N, D]
            # rel_vec_t: [B, D]
            struct_nodes_t, rel_vec_t, attn_over_nodes = self.ct_gap(
                temporal_state=h_t,
                node_repr=node_repr,
                adjacency_matrix=adj,
            )

            rel_vec_t = self.rel_proj(rel_vec_t)  # [B, H]

            # ----------------------------------------
            # 3.2 TRMB read / write
            # ----------------------------------------
            memory_context_t, memory_prob_t = self.trmb.read(rel_vec_t)
            self.trmb.write(rel_vec_t, memory_prob_t)

            # ----------------------------------------
            # 3.3 build aligned representation
            # target is question at t+1
            # ----------------------------------------
            next_q = question_ids[:, t + 1]                       # [B]
            next_concept = self.question_concept_map[next_q]     # [B]

            target_struct = self.gather_target_node_state(
                struct_nodes=struct_nodes_t,
                target_concept_ids=next_concept,
            )  # [B, D]

            if target_struct.size(-1) != self.hidden_dim:
                # safe projection if node_dim != hidden_dim
                target_struct = F.linear(
                    target_struct,
                    self.rel_proj.weight,
                    self.rel_proj.bias,
                )

            next_q_embed = self.question_embedding(next_q)  # [B, H]

            fused = torch.cat(
                [h_t, target_struct, memory_context_t, next_q_embed],
                dim=-1,
            )  # [B, 4H]

            z_t = self.fusion(fused)
            z_t = self.layer_norm(z_t)

            logit_t = self.predictor(z_t).squeeze(-1)  # [B]
            prob_t = torch.sigmoid(logit_t)

            logits_list.append(logit_t)
            probs_list.append(prob_t)
            y_true_list.append(responses[:, t + 1].float())
            valid_mask_list.append(mask[:, t + 1].float())

            rel_vector_list.append(rel_vec_t)
            memory_context_list.append(memory_context_t)
            memory_prob_list.append(memory_prob_t)

        # stack over time
        logits = torch.stack(logits_list, dim=1)                  # [B, T-1]
        probs = torch.stack(probs_list, dim=1)                    # [B, T-1]
        y_true = torch.stack(y_true_list, dim=1)                  # [B, T-1]
        valid_mask = torch.stack(valid_mask_list, dim=1)          # [B, T-1]

        rel_vectors = torch.stack(rel_vector_list, dim=1)         # [B, T-1, H]
        memory_contexts = torch.stack(memory_context_list, dim=1) # [B, T-1, H]
        memory_probs = torch.stack(memory_prob_list, dim=1)       # [B, T-1, M]

        # --------------------------------------------
        # loss
        # --------------------------------------------
        pred_loss = self.masked_bce_loss(logits, y_true, valid_mask)
        align_loss = self.masked_mse_loss(rel_vectors, memory_contexts, valid_mask)
        entropy_loss = self.masked_entropy_loss(memory_probs, valid_mask)

        total_loss = (
            pred_loss
            + self.lambda_align * align_loss
            + self.lambda_entropy * entropy_loss
        )

        loss_dict = {
            "total_loss": total_loss,
            "pred_loss": pred_loss,
            "align_loss": align_loss,
            "entropy_loss": entropy_loss,
        }

        output = {
            "logits": logits,
            "probs": probs,
            "y_true": y_true,
            "valid_mask": valid_mask,
            "loss_dict": loss_dict,
        }

        if return_details:
            output.update(
                {
                    "temporal_states": temporal_states,
                    "rel_vectors": rel_vectors,
                    "memory_contexts": memory_contexts,
                    "memory_probs": memory_probs,
                }
            )

        return output

    # ------------------------------------------------------------------
    # convenience methods
    # ------------------------------------------------------------------
    def predict(
        self,
        question_ids: torch.Tensor,
        responses: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Return probabilities only.
        """
        self.eval()
        with torch.no_grad():
            out = self.forward(question_ids, responses, mask=mask, return_details=False)
        return out["probs"]

    def compute_loss(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Expected batch keys:
            - question_ids
            - responses
            - mask (optional)
        """
        question_ids = batch["question_ids"]
        responses = batch["responses"]
        mask = batch.get("mask", None)

        out = self.forward(question_ids, responses, mask=mask, return_details=True)
        loss = out["loss_dict"]["total_loss"]
        return loss, out["loss_dict"]


def build_model_from_config(cfg) -> CTRANet:
    """
    Build model from config object or dict-like config.

    Required fields in cfg:
        num_questions
        num_concepts
        hidden_dim
        embed_dim
        node_dim
        memory_size
        dropout
        tau
        gamma
        memory_write_step
        lambda_align
        lambda_entropy

    Optional:
        adjacency_matrix
        question_concept_map
        use_concept_embedding
    """
    def get_attr(x, name, default=None):
        if isinstance(x, dict):
            return x.get(name, default)
        return getattr(x, name, default)

    model = CTRANet(
        num_questions=get_attr(cfg, "num_questions"),
        num_concepts=get_attr(cfg, "num_concepts"),
        hidden_dim=get_attr(cfg, "hidden_dim", 128),
        embed_dim=get_attr(cfg, "embed_dim", 128),
        node_dim=get_attr(cfg, "node_dim", 128),
        memory_size=get_attr(cfg, "memory_size", 64),
        dropout=get_attr(cfg, "dropout", 0.2),
        tau=get_attr(cfg, "tau", 0.5),
        gamma=get_attr(cfg, "gamma", 0.01),
        memory_write_step=get_attr(cfg, "memory_write_step", 0.5),
        lambda_align=get_attr(cfg, "lambda_align", 0.1),
        lambda_entropy=get_attr(cfg, "lambda_entropy", 0.01),
        adjacency_matrix=get_attr(cfg, "adjacency_matrix", None),
        question_concept_map=get_attr(cfg, "question_concept_map", None),
        use_concept_embedding=get_attr(cfg, "use_concept_embedding", True),
    )
    return model


if __name__ == "__main__":
    # simple sanity check
    B, T = 4, 20
    num_questions = 110
    num_concepts = 110

    q = torch.randint(0, num_questions, (B, T))
    r = torch.randint(0, 2, (B, T))
    m = torch.ones(B, T)

    adj = torch.eye(num_concepts)
    qc_map = torch.arange(num_questions) % num_concepts

    model = CTRANet(
        num_questions=num_questions,
        num_concepts=num_concepts,
        hidden_dim=128,
        embed_dim=128,
        node_dim=128,
        memory_size=64,
        adjacency_matrix=adj,
        question_concept_map=qc_map,
    )

    out = model(q, r, m)
    print("probs shape:", out["probs"].shape)  # [B, T-1]
    print("total loss:", out["loss_dict"]["total_loss"].item())