import GPy
import numpy as np
import pickle

class GPR():
    r"""
    This class adopts the Gaussian Process Regression for ROM applications. The map :math:`y = \mathcal{F}_{\boldsymbol{\theta}}(\mathbf{x})` is learnt. 
    
    Parameters
    ----------
    X : np.ndarray
        Input training data with dimension :math:`N_{samples} \times N_{features}`
    y : np.ndarray
        Output training data with dimension :math:`N_{samples}\times 1`
    kernel : GPy.Kernel (optional, default=None)
        Kernel for GPR
    """
    def __init__(self, X: np.ndarray, y: np.ndarray, 
                 kernel = None, normalisation = {'X': None, 'y': None}):
        
        self.n_samples, self.n_feature = X.shape
        assert self.n_samples == y.shape[0]
        
        if kernel is None:
            kernel = GPy.kern.RBF(input_dim=self.n_feature)
        
        # If required the data are rescaled
        data = [X, y]
        strings_data = list(normalisation.keys())
        
        rescaled_data = dict()
        self.norm_ = dict()
        for jj in range(len(strings_data)):
            self.norm_['which_'+strings_data[jj]] = normalisation[strings_data[jj]]  
            
            if normalisation[strings_data[jj]] == 'std':
                self.norm_['mean_' +strings_data[jj]] = np.mean(data[jj], axis=0)
                self.norm_['std_'  +strings_data[jj]] = np.std(data[jj],  axis=0)
                rescaled_data[strings_data[jj]] = self.transform(data[jj], strings_data[jj])
                
            elif normalisation[strings_data[jj]] == 'min-max':
                self.norm_['min_' +strings_data[jj]] = np.min(data[jj], axis=0)
                self.norm_['max_' +strings_data[jj]] = np.max(data[jj], axis=0)
                rescaled_data[strings_data[jj]] = self.transform(data[jj], strings_data[jj])
                
            else:
                rescaled_data[strings_data[jj]] = data[jj]
                
        self.train_X = rescaled_data['X']
        self.train_y = rescaled_data['y']
        
        self.model = GPy.models.GPRegression(self.train_X, self.train_y, kernel=kernel)
    
    def __call__(self, X_test: np.ndarray):
        return self.predict(X_test)[0].flatten()
    
    def optimize(self, optimizer='bfgs', messages = False, restart = None):
        r"""
        Optimise the GPR model.
        """
        if restart is None:
            self.model.optimize(optimizer=optimizer, messages=messages)
        else:
            self.model.optimize_restarts(optimizer=optimizer, num_restarts=restart, verbose=messages)
    
    def predict(self, X_test: np.ndarray):
        r"""
        Make new prediction for new values.
        
        Parameters
        ----------
        X : np.ndarray
            Input training data with dimension :math:`N_{s}^{test} \times N_{features}`
            
        Returns
        ----------
        
        mean : np.ndarray
        """
        
        if self.norm_['which_X'] is not None:
            X_test_rescaled = self.transform(X_test, 'X')
        else: 
            X_test_rescaled = X_test
            
        mean, cov =  self.model.predict(X_test_rescaled)
        
        if self.norm_['which_y'] is not None:
            mean_test_rescaled = self.inv_transform(mean, 'y')
            std_test_rescaled  = self.inv_transform_std(np.sqrt(cov), 'y')
            # std_test_rescaled = np.sqrt(cov)
        else: 
            mean_test_rescaled = mean
            std_test_rescaled = np.sqrt(cov)
        
        return mean_test_rescaled, std_test_rescaled
    
    def transform(self, data: np.ndarray, string_norm: str):
        r"""
        The data are forward rescaled according to the *min-max*
        
        .. math::
        \hat{x}_i(t) = \frac{x_i(t) - \min\limits_t x_i(t) }{\max\limits_t x_i(t) - \min\limits_t x_i(t)} \qquad \qquad i = 1, \dots, N_{feature}
        
        or *standardisation*
        
        .. math::
        \hat{x}_i(t) = \frac{x_i(t) -  \langle x_i\rangle_t }{\sigma_{x_i(t)}} \qquad \qquad i = 1, \dots, N_{feature}
        
        
        Parameters
        ----------
        data : np.ndarray
            Data to rescale.
        string_norm : string
            Label to identify the scaler type

        Returns
        -------
        norm_data : np.ndarray
            Rescaled data.
        """
        
        assert(self.norm_['which_'+string_norm] is not None)
        
        if self.norm_['which_'+string_norm] == 'min-max':
            min_scale = np.zeros_like(data)
            max_scale = np.zeros_like(data)
            
            for ii in range(len(self.norm_['min_'+string_norm])):
                min_scale[:, ii] = self.norm_['min_'+string_norm][ii]
                max_scale[:, ii] = self.norm_['max_'+string_norm][ii]
        
            return (data - min_scale) / (max_scale - min_scale)
        
        elif self.norm_['which_'+string_norm] == 'std':
            mean_scale = np.zeros_like(data)
            std_scale = np.zeros_like(data)
            
            for ii in range(len(self.norm_['mean_'+string_norm])):
                mean_scale[:, ii] = self.norm_['mean_'+string_norm][ii]
                std_scale[:, ii]  = self.norm_['std_'+string_norm][ii]
        
            return (data - mean_scale) / std_scale
        
    def inv_transform(self, norm_data: np.ndarray, string_norm: str):
      r"""
      The data are inverse rescaled according to the *min-max*
      
      .. math::
        {x}_i(t) = \min\limits_t x_i(t) + \hat{x}_i(t)\cdot \left(\max\limits_t x_i(t) - \min\limits_t x_i(t)\right) \qquad \qquad i = 1, \dots, N_{feature}
      
      or *standardisation*
      
      .. math::
        {x}_i(t) = \langle x_i\rangle_t  + \sigma_{x_i(t)}\cdot \hat{x}_i(t) \qquad \qquad i = 1, \dots, N_{feature}
      
      
      Parameters
      ----------
      norm_data : np.ndarray
        Data rescaled.
      string_norm : string
        Label to identify the scaler type

      Returns
      -------
      data : np.ndarray
        Inverse rescaled data.
      """
        
      assert(self.norm_['which_'+string_norm] is not None)
      
      if self.norm_['which_'+string_norm] == 'min-max':
          min_scale = np.zeros_like(norm_data)
          max_scale = np.zeros_like(norm_data)
          
          for ii in range(len(self.norm_['min_'+string_norm])):
              min_scale[:, ii] = self.norm_['min_'+string_norm][ii]
              max_scale[:, ii] = self.norm_['max_'+string_norm][ii]
      
          return min_scale + norm_data * (max_scale - min_scale)
      
      elif self.norm_['which_'+string_norm] == 'std':
          mean_scale = np.zeros_like(norm_data)
          std_scale = np.zeros_like(norm_data)
          
          for ii in range(len(self.norm_['mean_'+string_norm])):
              mean_scale[:, ii] = self.norm_['mean_'+string_norm][ii]
              std_scale[:, ii]  = self.norm_['std_'+string_norm][ii]
      
          return mean_scale + std_scale * norm_data
      
    def inv_transform_std(self, norm_data: np.ndarray, string_norm: str):
      r"""
      The data are inverse rescaled according to the *min-max*
      
      .. math::
        {x}_i(t) = \min\limits_t x_i(t) + \hat{x}_i(t)\cdot \left(\max\limits_t x_i(t) - \min\limits_t x_i(t)\right) \qquad \qquad i = 1, \dots, N_{feature}
      
      or *standardisation*
      
      .. math::
        {x}_i(t) = \langle x_i\rangle_t  + \sigma_{x_i(t)}\cdot \hat{x}_i(t) \qquad \qquad i = 1, \dots, N_{feature}
      
      
      The standard deviation of the prediction will be multiplied either by the denominator of each normlisation squared.
      
      Parameters
      ----------
      norm_data : np.ndarray
        Data rescaled.
      string_norm : string
        Label to identify the scaler type

      Returns
      -------
      data : np.ndarray
        Inverse rescaled data.
      """
        
      assert(self.norm_['which_'+string_norm] is not None)
      
      if self.norm_['which_'+string_norm] == 'min-max':
          min_scale = np.zeros_like(norm_data)
          max_scale = np.zeros_like(norm_data)
          
          for ii in range(len(self.norm_['min_'+string_norm])):
              min_scale[:, ii] = self.norm_['min_'+string_norm][ii]
              max_scale[:, ii] = self.norm_['max_'+string_norm][ii]
      
          return norm_data * (max_scale - min_scale)
      
      elif self.norm_['which_'+string_norm] == 'std':
          std_scale = np.zeros_like(norm_data)
          
          for ii in range(len(self.norm_['mean_'+string_norm])):
              std_scale[:, ii]  = self.norm_['std_'+string_norm][ii]
      
          return std_scale * norm_data

    def save(self, filename: str, store_train = False):
        r"""
        This function stores the model.
        """
        
        output = dict()
        
        output['norm_'] = self.norm_
        output['gpr_model'] = self.model
        
        if store_train:
            output['X_train'] = self.train_X
            output['y_train'] = self.train_y
        else:
            output['X_train'] = None
            output['y_train'] = None
            
        pickle.dump(output, open(filename, 'wb'))
        