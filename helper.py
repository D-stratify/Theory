import numpy as np
from scipy.interpolate import interp1d

def interp(Y, X, X_new, axis=1):
  """Interpolate the Chebyshev points onto a uniform grid."""
  Y_new = interp1d(X, Y, axis=axis, fill_value="extrapolate")
  return Y_new(X_new)

def time_average(Y, t, t0=0):
    """Time average for t>t0."""
    dom = t > t0
    return np.trapz(Y[dom,:,:], x=t[dom], axis=0)[0]/(t[-1]-t0)

class build_pdf:
    def __init__(self, Y1, Y2, bins=(150, 150), range_=None, f_X=None, df_X=None, dx=None):
        """
        Constructs a 2D PDF for Y1 and Y2
        f_X defines the sample space
        df_X defines the gradient of the sample space
        dx is the cell volume for the sample space

        The first dimension of Y1 and Y2 should correspond to samples or ensembles (e.g. (ns, nx1, nx2))
        """
        self.Y1 = Y1
        self.Y2 = Y2
        self.bins = bins

        # Number of spatial dimensions
        self.d = len(Y1.shape) - 1
        
        # Store sample normalised sample distributions
        self.dx = dx if dx is not None else (1,)*self.d
        self.f_X = f_X if f_X is not None else np.ones(Y1.shape[1:])
        self.df_X = df_X if f_X is not None else np.zeros(Y1.shape[1:])

        # Normalise
        mass = np.sum(self.f_X)*np.prod(self.dx)
        self.f_X = self.f_X/mass
        self.df_X = self.df_X/mass

        if range_ is None:
            self.range = [(self.Y1.min(), self.Y1.max()), (self.Y2.min(), self.Y2.max())]
        else:
            self.range = range_

        self.codim_X = (self.Y1.shape[0],)+(1,)*self.d
            
        ws = np.tile( self.f_X, (self.Y1.shape[0],)+(1,)*self.d ).flatten() if f_X is not None else None
            
        # Compute histogram and bin edges
        self.f_Y, self.y1edges, self.y2edges = np.histogram2d(self.Y1.flatten(), self.Y2.flatten(),
            bins=self.bins,
            range=self.range,
            density=True,
            weights=ws
        )

        # Compute cell centres
        self.y1 = self.y1edges[:-1] + np.diff(self.y1edges)/2
        self.y2 = self.y2edges[:-1] + np.diff(self.y2edges)/2

        self.y1_2d, self.y2_2d = np.meshgrid(self.y1, self.y2, indexing='ij')
        self.y1edges_2d, self.y2edges_2d = np.meshgrid(self.y1edges, self.y2edges, indexing='ij')
        
        # Cell size
        self.dy = (self.y1edges[1]-self.y1edges[0], self.y2edges[1]-self.y2edges[0])
        
        # Compute bin indices
        self.i1 = np.clip(np.digitize(self.Y1.flatten(), self.y1edges) - 1, 0, bins[0] - 1)
        self.i2 = np.clip(np.digitize(self.Y2.flatten(), self.y2edges) - 1, 0, bins[1] - 1)

        # Flatten 2D bin indices into 1D bin IDs
        self.flat_bin_idx = np.ravel_multi_index((self.i1, self.i2), dims=self.bins)

    def E_Y(self, Y3, boundary=True):
        """
        Computes conditional expectation of Y3 given bins of (Y1, Y2).
        Returns a (bins[0], bins[1]) array of conditional expectations.
        """
        Y3 = Y3.flatten()

        P = np.tile(self.f_X, self.codim_X).flatten()

        # New measure if boundary term
        if boundary:
          Q = np.tile(self.df_X, self.codim_X).flatten()
        else:
          Q = P
        
        bin_weight = np.bincount(self.flat_bin_idx, weights=P, minlength=np.prod(self.bins))
        bin_weighted_sum = np.bincount(self.flat_bin_idx, weights=Y3 * Q, minlength=np.prod(self.bins))

        # Avoid division by zero
        with np.errstate(invalid='ignore', divide='ignore'):
            bin_mean = bin_weighted_sum / bin_weight

        # Reshape to 2D grid
        return  np.nan_to_num(bin_mean.reshape(self.bins))

      
    def f_Y1(self):
        """Marginal distribution for Y1."""
        return np.sum(self.f_Y, axis=1) * self.dy[1]

    def f_Y2(self):
        """Marginal distribution for Y2."""
        return np.sum(self.f_Y, axis=0) * self.dy[0]

    def f_X1(self):
        """Marginal distribution for Y1."""
        return np.sum(self.f_X, axis=1) * self.dx[1]

    def f_X2(self):
        """Marginal distribution for Y2."""
        return np.sum(self.f_X, axis=0) * self.dx[0]
      
    def E_Y1(self, g_Y):
        """Expectation conditional on Y1."""
        return np.sum(self.f_Y*g_Y, axis=1)/self.f_Y1() * self.dy[1]

    def E_Y2(self, g_Y):
        """Expectation conditional on Y2."""
        return np.sum(self.f_Y*g_Y, axis=0)/self.f_Y2() * self.dy[0]
      
    def EY1_Y2(self):
        """Expectation of Y1 given Y2."""
        return np.sum(self.y1[:,np.newaxis]*self.f_Y, axis=0) / self.f_Y2() * self.dy[0]

    def EY2_Y1(self):
        """Expectation of Y2 given Y1."""
        return np.sum(self.y2[np.newaxis,:]*self.f_Y, axis=1) / self.f_Y1() * self.dy[1]
    
    def f_Y2(self):
        """Marginal distribution for Y2."""
        return np.sum(self.f_Y, axis=0) * self.dy[0] 
    
    def E(self, g):
        """
        Computes the expectation of g
        """
        return np.sum(np.sum(self.f_Y*g, axis=0), axis=0) * np.prod(self.dy)

    
