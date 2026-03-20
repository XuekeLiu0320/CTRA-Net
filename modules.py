# modules.py
# -*- coding: utf-8 -*-

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class InteractionEmbedding(nn.Module):
    """
    Embed (question_id, response) into a unified interaction representation.

    Common KT practice:
    - question_id in [0, num_questions - 1]
    - response in {0, 1}
    - interaction index = question_id * 2 + response
    """

    def __init__(
        self,
        num_questions: int,
        embed_dim: int,
        padding_idx: int = 0
    ) -> None:
        super().__init__()
        self.num_questions = num_questions
        self.embed_dim = embed_dim
        self.padding_idx = padding_idx

        # question embedding
        self.question_emb = nn.Embedding(
            num_embeddings=num_questions + 1,
            embedding_dim=embed_dim,
            padding_idx=padding_idx
        )

        # interaction embedding: 2 * num_questions + 1 for padding
        self.interaction_emb = nn.Embedding(
            num_embeddings=2 * num_questions + 1,
            embedding_dim=embed_dim,
            padding_idx=padding_idx
        )

        self.layer_norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(0.2)

    def forward(
        self,
        question_ids: torch.Tensor,
        responses: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            question_ids: [B, T]
            responses:    [B, T], values in {0,1}
        Returns:
            interaction_repr: [B, T, D]
            question_repr:    [B, T, D]
        """
        question_ids = question_ids.long()
        responses = responses.long().clamp(min=0, max=1)

        # padding positions stay 0
        interaction_ids = question_ids * 2 + responses
        interaction_ids = torch.where(
            question_ids == self.padding_idx,
            torch.zeros_like(interaction_ids),
            interaction_ids
        )

        q_emb = self.question_emb(question_ids)               # [B, T, D]
        inter_emb = self.interaction_emb(interaction_ids)     # [B, T, D]

        x = q_emb + inter_emb
        x = self.layer_norm(x)
        x = self.dropout(x)
        return x, q_emb


class CTGAP(nn.Module):
    """
    Cross-Temporal Graph Attention Propagation

    Paper idea:
    - Current temporal state h_t acts as query
    - Graph node representations act as keys/values
    - Generate time-conditioned relation weights
    - Inject attention into base adjacency matrix
    - Perform dynamic graph propagation
    """

    def __init__(
        self,
        hidden_dim: int,
        node_dim: int,
        graph_out_dim: int,
        dropout: float = 0.2,
        activation: str = "relu"
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.node_dim = node_dim
        self.graph_out_dim = graph_out_dim

        self.query_proj = nn.Linear(hidden_dim, graph_out_dim)
        self.key_proj = nn.Linear(node_dim, graph_out_dim)
        self.value_proj = nn.Linear(node_dim, graph_out_dim)

        self.out_proj = nn.Linear(graph_out_dim, graph_out_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(graph_out_dim)

        if activation == "relu":
            self.act = nn.ReLU()
        elif activation == "gelu":
            self.act = nn.GELU()
        else:
            raise ValueError(f"Unsupported activation: {activation}")

    @staticmethod
    def normalize_adj(adj: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        """
        Symmetric normalization: D^{-1/2} A D^{-1/2}

        Args:
            adj: [N, N]
        Returns:
            norm_adj: [N, N]
        """
        degree = adj.sum(dim=-1)  # [N]
        degree_inv_sqrt = torch.pow(degree + eps, -0.5)
        d_mat = torch.diag(degree_inv_sqrt)
        return d_mat @ adj @ d_mat

    def forward(
        self,
        h_t: torch.Tensor,
        node_repr: torch.Tensor,
        base_adj: torch.Tensor,
        node_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            h_t:       [B, H]
            node_repr: [N, Dn]
            base_adj:  [N, N], base item/knowledge adjacency
            node_mask: [N], optional binary mask
        Returns:
            propagated_summary: [B, G]
            relation_vector:    [B, G]
            updated_nodes:      [B, N, G]
        """
        bsz = h_t.size(0)
        num_nodes = node_repr.size(0)

        # projections
        q = self.query_proj(h_t)                     # [B, G]
        k = self.key_proj(node_repr)                 # [N, G]
        v = self.value_proj(node_repr)               # [N, G]

        # attention logits: [B, N]
        attn_logits = torch.matmul(q, k.transpose(0, 1)) / (self.graph_out_dim ** 0.5)

        if node_mask is not None:
            mask = node_mask.unsqueeze(0).expand(bsz, -1)  # [B, N]
            attn_logits = attn_logits.masked_fill(mask == 0, float("-inf"))

        alpha = F.softmax(attn_logits, dim=-1)      # [B, N]
        alpha = self.dropout(alpha)

        # relation modulation vector r_t = sum(alpha * value)
        relation_vector = torch.matmul(alpha, v)     # [B, G]

        # dynamic adjacency for each sample
        # paper intuition: A_t = A \odot (alpha 1^T)
        # here we broadcast node-wise attention over rows
        dynamic_adj = alpha.unsqueeze(-1) * base_adj.unsqueeze(0)   # [B, N, N]

        # normalized propagation
        updated_nodes = []
        for i in range(bsz):
            a_i = dynamic_adj[i] + torch.eye(num_nodes, device=base_adj.device, dtype=base_adj.dtype)
            a_i = self.normalize_adj(a_i)
            z_i = torch.matmul(a_i, v)              # [N, G]
            z_i = self.out_proj(z_i)
            z_i = self.act(z_i)
            z_i = self.layer_norm(z_i)
            updated_nodes.append(z_i)

        updated_nodes = torch.stack(updated_nodes, dim=0)  # [B, N, G]

        # graph summary under temporal condition
        propagated_summary = torch.sum(alpha.unsqueeze(-1) * updated_nodes, dim=1)  # [B, G]

        return propagated_summary, relation_vector, updated_nodes


class TRMB(nn.Module):
    """
    Temporal-Relational Memory Bank

    Memory is maintained per batch item during a forward pass.
    This module supports:
    - projection + normalization of relation vector
    - similarity matching
    - time-decay reweighting
    - memory read
    - gated memory update
    """

    def __init__(
        self,
        input_dim: int,
        memory_dim: int,
        memory_size: int = 64,
        write_rate: float = 0.5,
        decay_lambda: float = 0.01,
        temperature: float = 0.5,
        eps: float = 1e-8
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.memory_dim = memory_dim
        self.memory_size = memory_size
        self.write_rate = write_rate
        self.decay_lambda = decay_lambda
        self.temperature = temperature
        self.eps = eps

        self.write_proj = nn.Linear(input_dim, memory_dim)
        self.read_proj = nn.Linear(memory_dim, memory_dim)
        self.layer_norm = nn.LayerNorm(memory_dim)

    def init_memory(
        self,
        batch_size: int,
        device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            memory_bank: [B, M, D]
            timestamps:  [B, M]
        """
        memory_bank = torch.zeros(batch_size, self.memory_size, self.memory_dim, device=device)
        timestamps = torch.zeros(batch_size, self.memory_size, device=device)
        return memory_bank, timestamps

    def project_relation(self, relation_vector: torch.Tensor) -> torch.Tensor:
        """
        Eq.(5)-style normalized memory write vector.
        Args:
            relation_vector: [B, Din]
        Returns:
            z_t: [B, Dm]
        """
        z_t = self.write_proj(relation_vector)
        z_t = z_t / (torch.norm(z_t, p=2, dim=-1, keepdim=True) + self.eps)
        return z_t

    def compute_matching(
        self,
        z_t: torch.Tensor,
        memory_bank: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute similarity and soft matching.
        Args:
            z_t:        [B, D]
            memory_bank:[B, M, D]
        Returns:
            logits:     [B, M]
            weights:    [B, M]
        """
        logits = torch.sum(memory_bank * z_t.unsqueeze(1), dim=-1) / self.temperature  # [B, M]
        weights = F.softmax(logits, dim=-1)
        return logits, weights

    def apply_time_decay(
        self,
        weights: torch.Tensor,
        timestamps: torch.Tensor,
        current_step: int
    ) -> torch.Tensor:
        """
        Eq.(8)-style time decay reweighting.
        Args:
            weights:     [B, M]
            timestamps:  [B, M]
            current_step: int
        Returns:
            decayed_weights: [B, M]
        """
        delta_t = float(current_step) - timestamps
        decay = torch.exp(-self.decay_lambda * delta_t).clamp(min=self.eps)  # [B, M]
        decayed = weights * decay
        decayed = decayed / (decayed.sum(dim=-1, keepdim=True) + self.eps)
        return decayed

    def read(
        self,
        decayed_weights: torch.Tensor,
        memory_bank: torch.Tensor
    ) -> torch.Tensor:
        """
        Eq.(7)-style memory retrieval.
        Args:
            decayed_weights: [B, M]
            memory_bank:     [B, M, D]
        Returns:
            context:         [B, D]
        """
        context = torch.sum(decayed_weights.unsqueeze(-1) * memory_bank, dim=1)
        context = self.read_proj(context)
        context = self.layer_norm(context)
        return context

    def write(
        self,
        z_t: torch.Tensor,
        decayed_weights: torch.Tensor,
        memory_bank: torch.Tensor,
        timestamps: torch.Tensor,
        current_step: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Eq.(9)-style gated update.
        Args:
            z_t:             [B, D]
            decayed_weights: [B, M]
            memory_bank:     [B, M, D]
            timestamps:      [B, M]
            current_step:    int
        Returns:
            new_memory_bank: [B, M, D]
            new_timestamps:  [B, M]
        """
        update_term = decayed_weights.unsqueeze(-1) * z_t.unsqueeze(1)      # [B, M, D]
        keep_term = (1.0 - self.write_rate * decayed_weights.unsqueeze(-1)) * memory_bank
        write_term = self.write_rate * update_term

        new_memory_bank = keep_term + write_term
        new_memory_bank = new_memory_bank / (
            torch.norm(new_memory_bank, p=2, dim=-1, keepdim=True) + self.eps
        )

        # softly refresh timestamps according to attention
        current_step_tensor = torch.full_like(timestamps, float(current_step))
        new_timestamps = (1.0 - decayed_weights) * timestamps + decayed_weights * current_step_tensor

        return new_memory_bank, new_timestamps

    def forward(
        self,
        relation_vector: torch.Tensor,
        memory_bank: torch.Tensor,
        timestamps: torch.Tensor,
        current_step: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Full TRMB step.

        Args:
            relation_vector: [B, Din]
            memory_bank:     [B, M, Dm]
            timestamps:      [B, M]
            current_step:    int

        Returns:
            context:         [B, Dm]
            z_t:             [B, Dm]
            decayed_weights: [B, M]
            new_memory_bank: [B, M, Dm]
            new_timestamps:  [B, M]
        """
        z_t = self.project_relation(relation_vector)                            # [B, Dm]
        _, weights = self.compute_matching(z_t, memory_bank)                    # [B, M]
        decayed_weights = self.apply_time_decay(weights, timestamps, current_step)
        context = self.read(decayed_weights, memory_bank)
        new_memory_bank, new_timestamps = self.write(
            z_t, decayed_weights, memory_bank, timestamps, current_step
        )
        return context, z_t, decayed_weights, new_memory_bank, new_timestamps


class FusionPredictor(nn.Module):
    """
    Fuse temporal state, graph summary, and memory context, then predict correctness.
    """

    def __init__(
        self,
        hidden_dim: int,
        graph_dim: int,
        memory_dim: int,
        fusion_dim: int,
        dropout: float = 0.2
    ) -> None:
        super().__init__()
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim + graph_dim + memory_dim, fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(fusion_dim)
        )
        self.classifier = nn.Linear(fusion_dim, 1)

    def forward(
        self,
        h_t: torch.Tensor,
        graph_summary: torch.Tensor,
        memory_context: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            h_t:            [B, H]
            graph_summary:  [B, G]
            memory_context: [B, M]
        Returns:
            logits:         [B]
            fused_repr:     [B, F]
        """
        fused = torch.cat([h_t, graph_summary, memory_context], dim=-1)
        fused_repr = self.fusion(fused)
        logits = self.classifier(fused_repr).squeeze(-1)
        return logits, fused_repr


class RelationalAlignmentLoss(nn.Module):
    """
    L_align = || r_t - c_t ||_2
    """

    def __init__(self, reduction: str = "mean") -> None:
        super().__init__()
        self.reduction = reduction

    def forward(
        self,
        relation_vector: torch.Tensor,
        memory_context: torch.Tensor
    ) -> torch.Tensor:
        loss = torch.norm(relation_vector - memory_context, p=2, dim=-1)
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


class EntropyRegularization(nn.Module):
    """
    L_ent = - sum p log p
    Usually added as a regularizer term.
    """

    def __init__(self, reduction: str = "mean", eps: float = 1e-8) -> None:
        super().__init__()
        self.reduction = reduction
        self.eps = eps

    def forward(self, probs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            probs: [B, M]
        """
        ent = -torch.sum(probs * torch.log(probs + self.eps), dim=-1)
        if self.reduction == "mean":
            return ent.mean()
        if self.reduction == "sum":
            return ent.sum()
        return ent