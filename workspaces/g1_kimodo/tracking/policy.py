"""Preserve the validated balance actor while PPO learns motion-specific residuals."""
import copy
import torch
from rsl_rl.modules import ActorCritic


class ResidualActor(torch.nn.Module):
    def __init__(self,base,residual):
        super().__init__();self.base=base;self.residual=residual
    def forward(self,x):return self.base(x)+self.residual(x)


class TrackingActorCritic(ActorCritic):
    def __init__(self,*args,base_checkpoint,**kwargs):
        super().__init__(*args,**kwargs)
        base=copy.deepcopy(self.actor)
        weights=torch.load(base_checkpoint,map_location='cpu',weights_only=False)['model_state_dict']
        base.load_state_dict({k.removeprefix('actor.'):v for k,v in weights.items() if k.startswith('actor.')})
        base.requires_grad_(False)
        with torch.no_grad():self.actor[-1].weight.zero_();self.actor[-1].bias.zero_()
        self.actor=ResidualActor(base,self.actor)
        print('Actor: frozen validated balance MLP + trainable zero-output residual MLP',flush=True)
