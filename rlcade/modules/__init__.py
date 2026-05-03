from rlcade.modules.encoders import (
    CNNEncoder,
    LSTMEncoder,
    ResNetEncoder,
    create_encoder,
    build_encoder_kwargs,
    parse_channels,
)
from rlcade.modules.heads import (
    PolicyHead,
    ValueHead,
    DuelingHead,
    DistributionalDuelingHead,
    NoisyLinear,
)
from rlcade.modules.actor import Actor
from rlcade.modules.critic import Critic
from rlcade.modules.lstm import LstmActorCritic
from rlcade.modules.qnet import QNet, RainbowQNet

_ACTORS = {
    "actor": Actor,
}

_CRITICS = {
    "critic": Critic,
}

_QNETS = {
    "qnet": QNet,
    "rainbow_qnet": RainbowQNet,
}


def create_actor(name: str, *args, **kwargs):
    """Create an actor by name. Raises KeyError if unknown."""
    return _ACTORS[name](*args, **kwargs)


def create_critic(name: str, *args, **kwargs):
    """Create a critic by name. Raises KeyError if unknown."""
    return _CRITICS[name](*args, **kwargs)


def create_qnet(name: str, *args, **kwargs):
    """Create a Q-network by name. Raises KeyError if unknown."""
    return _QNETS[name](*args, **kwargs)
