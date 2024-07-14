import gpytorch
import torch

class BatchIndependentMultitaskGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, nout):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ZeroMean(batch_shape=torch.Size([nout]))
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(batch_shape=torch.Size([nout])),
            batch_shape=torch.Size([nout])
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal.from_batch_mvn(
            gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
        )


class MultitaskGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, nout, rank=None):
        '''
        Parameters
        ----------
        nout : int
            number of outputs
        '''
        super(MultitaskGPModel, self).__init__(train_x, train_y, likelihood)

        if rank is None:
            rank = nout

        self.mean_module = gpytorch.means.MultitaskMean(
            # gpytorch.means.ConstantMean(), num_tasks=nout
            gpytorch.means.ZeroMean(), num_tasks=nout
        )
        self.covar_module = gpytorch.kernels.MultitaskKernel(
            gpytorch.kernels.RBFKernel(ard_num_dims=train_x.shape[1]), num_tasks=nout, rank=rank
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)

class IndependentGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        '''
        Parameters
        ----------
        nout : int
            number of outputs
        '''
        super(IndependentGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu = 1.5, ard_num_dims=train_x.shape[1]))
        # self.covar_module = gpytorch.kernels.MaternKernel(nu = 1.5, ard_num_dims=train_x.shape[1])
        # self.covar_module = gpytorch.kernels.RBFKernel(ard_num_dims=train_x.shape[1])

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
    
class BatchMultitaskGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, batch_shape):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean(
            batch_shape=batch_shape
        )  # (prior=gpytorch.priors.NormalPrior(4.9132,0.01))
        self.base_kernel = gpytorch.kernels.RBFKernel(
            ard_num_dims=3, batch_shape=batch_shape
        )
        self.covar_module = gpytorch.kernels.ScaleKernel(
            self.base_kernel, batch_shape=batch_shape
        )
        self.batch_shape = batch_shape

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)
    
class BatchExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood,batch_shape):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean(batch_shape=batch_shape)
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(batch_shape=batch_shape), 
            batch_shape = batch_shape)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)



# FROM HERE ONWARD: Functions added specifically for nonparamatric learning MPC
# Author: Mathieu Dubied (mdubied@ethz.ch)
# Date: 17/06/2024

# Define the GP batch GP model
class BatchGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, batch_shape):
        super(BatchGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean(batch_shape=batch_shape)
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(batch_shape=batch_shape),
            batch_shape=batch_shape
        )


    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
