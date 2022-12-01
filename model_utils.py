from torch import nn

# Select activation function from hyperparams inputs
class ActivationSwitch:
    def fn(self, activation):
        default = nn.PReLU()
        return getattr(
            self, f"activation_function_{activation.lower()}", lambda: default
        )()

    def activation_function_relu(self):
        return nn.ReLU()

    def activation_function_prelu(self):
        return nn.PReLU()

    def activation_function_tanh(self):
        return nn.Tanh()

    def activation_function_sigmoid(self):
        return nn.Sigmoid()

    def activation_function_elu(self):
        return nn.ELU()

    def activation_function_leakyrelu(self):
        return nn.LeakyReLU()

    def activation_function_selu(self):
        return nn.SELU()
