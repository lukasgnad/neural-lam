from neural_lam.models.ar_model import ARModel
import torch

class PersistenceBaseline(ARModel):
    
    def predict_step(self, prev_state, prev_prev_state, forcing):
        """
        Persistence baseline: X_{t+1} = X_t
        Ignores prev_prev_state and forcing.
        
        Inputs:
            prev_state: (B, num_grid_nodes, feature_dim) = X_t
            prev_prev_state: (B, num_grid_nodes, feature_dim) = X_{t-1}
            forcing: (B, num_grid_nodes, forcing_dim)

        Returns:
            predicted_state: same as prev_state
            predicted_std: None
        """
        # Directly return the previous state as the prediction
        return prev_state, None
    
    def common_step(self, batch):
        """
        Predict on single batch
        batch consists of:
        init_states: (B, 2, num_grid_nodes, d_features)
        target_states: (B, pred_steps, num_grid_nodes, d_features)
        forcing_features: (B, pred_steps, num_grid_nodes, d_forcing),
            where index 0 corresponds to index 1 of init_states
        """
        (
            init_states,
            target_states,
            forcing_features,
            all_init_states
        ) = batch

        prediction, pred_std = self.unroll_prediction(
            init_states, forcing_features, target_states, all_init_states
        )
        return prediction, target_states, pred_std
        
    def unroll_prediction(self, init_states, forcing_features, true_states, all_init_states):
        """
        Roll out prediction taking multiple autoregressive steps with model
        init_states: (B, 2, num_grid_nodes, d_f)
        forcing_features: (B, pred_steps, num_grid_nodes, d_static_f)
        true_states: (B, pred_steps, num_grid_nodes, d_f)
        """
        
        # prev_prev_state = init_states[:, 0]
        # prev_state = init_states[:, 1]
        prediction_list = []
        pred_std_list = []
        pred_steps = forcing_features.shape[1]

        for i in range(pred_steps):
            forcing = forcing_features[:, i]
            border_state = true_states[:, i]

            pred_state, pred_std = all_init_states[:, i%4], None
            # state: (B, num_grid_nodes, d_f)
            # pred_std: (B, num_grid_nodes, d_f) or None

            new_state = self.optional_boundary_forcing(pred_state, border_state)

            prediction_list.append(new_state)
            if self.output_std:
                pred_std_list.append(pred_std)

            # Update conditioning states
            # prev_prev_state = prev_state
            # prev_state = new_state

        prediction = torch.stack(
            prediction_list, dim=1
        )  # (B, pred_steps, num_grid_nodes, d_f)
        if self.output_std:
            pred_std = torch.stack(
                pred_std_list, dim=1
            )  # (B, pred_steps, num_grid_nodes, d_f)
        else:
            pred_std = self.per_var_std  # (d_f,)

        return prediction, pred_std