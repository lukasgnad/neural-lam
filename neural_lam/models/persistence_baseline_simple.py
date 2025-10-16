from neural_lam.models.ar_model import ARModel
import torch

class PersistenceBaselineSimple(ARModel):
    
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