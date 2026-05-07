class SDSPruning(nn.Module):
    def __init__(self, feat_dim=1024, prune_ratio=0.7, token_batch=128, pivot_num=1):
        super(SDSPruning, self).__init__()
        self.feat_dim = feat_dim
        self.prune_ratio = prune_ratio
        self.token_batch = token_batch
        self.pivot_num = pivot_num
        self.beta = 0.3

    def forward(self, x, pos):

        B, N, D = x.shape
        device = x.device

        keep_num = int(N * (1 - self.prune_ratio))
        keep_num = max(keep_num, self.pivot_num + 1)

        batch_kept_x = []
        batch_kept_pos = []

        for b in range(B):
            x_b = x[b]
            pos_b = pos[b].float()

            x_b_norm = F.normalize(x_b, p=2, dim=-1)

            coord_min = pos_b.min(dim=0).values
            coord_max = pos_b.max(dim=0).values
            scale_factor = 1.0 / (torch.norm(coord_max - coord_min) + 1e-6)

            norm_l1 = torch.norm(x_b, p=1, dim=-1)
            first_pivot = torch.argmax(norm_l1).item()

            selected_indices = [first_pivot]

            candidate_mask = torch.ones(N, dtype=torch.bool, device=device)
            candidate_mask[first_pivot] = False

            first_pivot_feat = x_b_norm[first_pivot].unsqueeze(0)
            first_pivot_pos = pos_b[first_pivot].unsqueeze(0)

            current_max_sim = torch.mm(x_b_norm, first_pivot_feat.T).squeeze()

            current_min_dist = torch.cdist(pos_b, first_pivot_pos).squeeze() * scale_factor

            while len(selected_indices) < keep_num:

                bss_sim = current_max_sim * (1.0 + 0.5 * current_min_dist)

                bss_sim_masked = bss_sim.clone()
                bss_sim_masked[~candidate_mask] = float('inf')

                current_batch_size = min(self.token_batch, keep_num - len(selected_indices))

                _, batch_new_indices = torch.topk(
                    bss_sim_masked,
                    k=current_batch_size,
                    largest=False
                )

                batch_new_indices_list = batch_new_indices.tolist()
                selected_indices.extend(batch_new_indices_list)
                candidate_mask[batch_new_indices] = False

                if len(selected_indices) >= keep_num:
                    break

                new_feats = x_b_norm[batch_new_indices]
                new_pos = pos_b[batch_new_indices]

                new_sims = torch.mm(x_b_norm, new_feats.T)

                current_max_sim = torch.max(
                    current_max_sim,
                    new_sims.max(dim=1).values
                )

                new_dists = torch.cdist(pos_b, new_pos) * scale_factor

                current_min_dist = torch.min(
                    current_min_dist,
                    new_dists.min(dim=1).values
                )

            kept_indices_tensor = torch.tensor(selected_indices, device=device)

            final_kept_x = x_b[kept_indices_tensor]

            discarded_indices = torch.nonzero(candidate_mask).squeeze()

            if len(discarded_indices) > 0:

                disc_x = x_b_norm[discarded_indices]
                kept_x = x_b_norm[kept_indices_tensor]

                chunk_size = 512

                delta_features = torch.zeros_like(final_kept_x)

                weight_accumulator = torch.zeros(
                    len(selected_indices),
                    1,
                    device=device
                )

                for i in range(0, len(discarded_indices), chunk_size):

                    chunk_disc = disc_x[i:i + chunk_size]

                    chunk_idx = discarded_indices[i:i + chunk_size]

                    sim_chunk = torch.mm(chunk_disc, kept_x.T)

                    max_val, max_idx = sim_chunk.max(dim=1)

                    weighted_feat = x_b[chunk_idx] * max_val.unsqueeze(1)

                    delta_features.index_add_(
                        0,
                        max_idx,
                        weighted_feat
                    )

                    weight_accumulator.index_add_(
                        0,
                        max_idx,
                        max_val.unsqueeze(1)
                    )

                mask_has_aggregation = (
                    weight_accumulator > 1e-8
                ).squeeze()

                delta_features[mask_has_aggregation] /= (
                    weight_accumulator[mask_has_aggregation]
                )

                final_kept_x[mask_has_aggregation] = (
                    self.beta * final_kept_x[mask_has_aggregation] +
                    (1 - self.beta) * delta_features[mask_has_aggregation]
                )

            batch_kept_x.append(final_kept_x)

            batch_kept_pos.append(pos_b[kept_indices_tensor])

        x_out = torch.stack(batch_kept_x)

        pos_out = torch.stack(batch_kept_pos)

        global_feat = x.mean(dim=1)

        preds = torch.zeros(B, 1).to(device)

        return x_out, pos_out, preds