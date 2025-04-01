from pyknos.nflows.flows import Flow
from nflows.flows.base import Flow
from nflows.distributions.normal import StandardNormal
from nflows.transforms.base import CompositeTransform
from nflows.transforms.autoregressive import MaskedAffineAutoregressiveTransform
from nflows.transforms.permutations import ReversePermutation
import torch
from sbi import utils as sbi_utils

class PriorFiltered():
    """Class for creating a prior distribution from
       heuristically filtered simulations"""
    
    def __init__(self, parameters):
        self.parameters = parameters
        nparams = len(parameters)
        self.flow = self.__flow_init(nparams)
        self.acc_rate = 0.0

    def __flow_init(self, nparams):
        num_layers = 5
        base_dist = StandardNormal(shape=[nparams])
        transforms = []
        for _ in range(num_layers):
            transforms.append(ReversePermutation(features=nparams))
            transforms.append(MaskedAffineAutoregressiveTransform(features=nparams,
                                hidden_features=50,
                                context_features=None,
                                num_blocks=2,
                                use_residual_blocks=False,
                                random_mask=False,
                                activation=torch.tanh,
                                dropout_probability=0.0,
                                use_batch_norm=True))
        transform = CompositeTransform(transforms)
        return Flow(transform, base_dist)

    def sample(self, sample_shape, return_acc_rate=False):
        base_prior = UniformPrior(self.parameters)        
        nsamples = sample_shape[0]
        nsamples_keep = 0
        samples_keep = torch.tensor([])
        total_samples = 0
        while nsamples_keep < nsamples:
            samples = self.flow.sample(nsamples).detach()
            total_samples = total_samples + len(samples)
            indices = list()
            for idx in range(len(samples)):
                try:
                    if base_prior.log_prob(samples[idx]) == 0:
                        indices.append(idx)
                except ValueError:
                    pass
                
            samples_keep = torch.cat([samples_keep, samples[indices]])
            nsamples_keep = len(samples_keep)
        acc_rate = torch.tensor(nsamples_keep / total_samples)
        samples = samples_keep[:nsamples]   
        if return_acc_rate:
            return samples, acc_rate
        else:
            return samples     

    def log_prob(self, theta):
        if self.acc_rate == 0.0:
            # get the acceptance rate right away
            _, acc_rate = self.sample((10_000,), return_acc_rate=True)            
        return self.flow.log_prob(theta).detach() - torch.log(acc_rate)
    
class Flow_base(Flow):
    def __init__(self, batch_theta, batch_x, embedding_net, n_layers=5,
                 z_score_theta=True, z_score_x=True, device='cpu'):

        # instantiate the flow
        flow = build_nsf(batch_x=batch_theta,
                         batch_y=batch_x,
                         z_score_x=z_score_theta,
                         z_score_y=z_score_x,
                         embedding_net=embedding_net).to(device)

        super().__init__(flow._transform, 
                         flow._distribution, 
                         flow._embedding_net)

    def save_state(self, filename):
        state_dict = {}
        state_dict['flow'] = self.state_dict()
        torch.save(state_dict, filename)

    def load_state(self, filename):
        state_dict = torch.load(filename, map_location=device)
        self.load_state_dict(state_dict['flow'])

def build_flow(batch_theta,
               batch_x,
               embedding_net,
               **kwargs):

    flow = Flow_base(batch_theta, 
                     batch_x, 
                     embedding_net,
                     device,
                     **kwargs).to(device)

    return flow

class UniformPrior(sbi_utils.BoxUniform):
    """Prior distribution object that generates uniform sample on range (0,1)"""
    def __init__(self, parameters):
        """
        Parameters
        ----------
        parameters: list of str
            List of parameter names for prior distribution
        """
        self.parameters = parameters
        low = len(parameters)*[0]
        high = len(parameters)*[1]
        super().__init__(low=torch.tensor(low, dtype=torch.float32),
                         high=torch.tensor(high, dtype=torch.float32))